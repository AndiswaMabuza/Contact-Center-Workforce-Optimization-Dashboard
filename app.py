# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

# --- Configuration ---
st.set_page_config(
    page_title="Contact Center Workforce Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data (with caching for performance) ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Ensure datetime columns are correctly parsed after loading from CSV
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['ScheduledShiftStart'] = pd.to_datetime(df['ScheduledShiftStart'])
    df['ScheduledShiftEnd'] = pd.to_datetime(df['ScheduledShiftEnd'])
    df['ActualLogin'] = pd.to_datetime(df['ActualLogin'])
    df['ActualLogout'] = pd.to_datetime(df['ActualLogout'])
    df['EventStartTime'] = pd.to_datetime(df['EventStartTime'])
    df['EventEndTime'] = pd.to_datetime(df['EventEndTime'])
    return df

df = load_data('agent_performance_data.csv')

# --- Sidebar Filters ---
st.sidebar.header("Filter Data")

min_date = df['Date'].min()
max_date = df['Date'].max()

# --- NEW: Quick Date Filter Selectbox ---
quick_filter_options = ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
quick_filter = st.sidebar.selectbox("Quick Filter", options=quick_filter_options)

# Calculate start and end dates based on the quick filter selection
if quick_filter == "Last 7 Days":
    start_date_for_picker = max_date - datetime.timedelta(days=6)
    end_date_for_picker = max_date
elif quick_filter == "Last 30 Days":
    start_date_for_picker = max_date - datetime.timedelta(days=29)
    end_date_for_picker = max_date
elif quick_filter == "Last 90 Days":
    start_date_for_picker = max_date - datetime.timedelta(days=89)
    end_date_for_picker = max_date
else: # "All Time"
    start_date_for_picker = min_date
    end_date_for_picker = max_date

# --- Manual Date Range Override ---
date_range = st.sidebar.date_input(
    "Custom Date Range",
    value=(start_date_for_picker, end_date_for_picker), # Default to the quick filter's range
    min_value=min_date,
    max_value=max_date
)

# Initialize filtered_df here with a default
filtered_df = df.copy()

# Apply the date filter based on the manual date picker
if len(date_range) == 2:
    start_date_filter = date_range[0]
    end_date_filter = date_range[1]
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date_filter) & (filtered_df['Date'] <= end_date_filter)].copy()
else:
    st.sidebar.warning("Please select a complete date range.")
    st.stop()


# Agent Selection Filter (applied to the date-filtered data)
available_agents = sorted(filtered_df['AgentID'].unique().tolist())
selected_agents = st.sidebar.multiselect(
    "Select Agents (Optional)",
    options=available_agents,
    default=available_agents
)

if selected_agents:
    filtered_df = filtered_df[filtered_df['AgentID'].isin(selected_agents)].copy()
else:
    pass

# --- Debug Filter Status ---
with st.sidebar.expander("Debug Filter Status"):
    st.write(f"Quick Filter: {quick_filter}")
    st.write(f"Manual Date Input: {date_range}")
    st.write(f"Filtered Date Range: {start_date_filter} to {end_date_filter}")
    st.write(f"Selected Agents: {selected_agents if selected_agents else 'All Agents'}")
    st.write(f"Original Data Shape (rows, cols): {df.shape}")
    st.write(f"Filtered Data Shape (rows, cols): {filtered_df.shape}")
    if not filtered_df.empty:
        st.write(f"Unique Dates in Filtered Data: {filtered_df['Date'].nunique()}")
        st.write(f"Unique Agents in Filtered Data: {filtered_df['AgentID'].nunique()}")


# Check if filtered_df is empty after all selections
if filtered_df.empty:
    st.error("No data available for the selected filters. Please adjust your selections.")
    st.stop()

# --- Data Processing and Calculations (as in notebook cells 3-6) ---
@st.cache_data
def calculate_adherence(data_frame):
    if data_frame.empty:
        return pd.DataFrame(columns=['AgentID', 'Date', 'ScheduledShiftStart', 'ScheduledShiftEnd',
                                     'ActualLogin', 'ActualLogout', 'ScheduledDurationMinutes',
                                     'ActualDurationMinutes', 'AdherencePercentage'])

    agent_daily_shifts = data_frame.groupby(['AgentID', 'Date']).agg(
        ScheduledShiftStart=('ScheduledShiftStart', 'first'),
        ScheduledShiftEnd=('ScheduledShiftEnd', 'first'),
        ActualLogin=('ActualLogin', 'first'),
        ActualLogout=('ActualLogout', 'first')
    ).reset_index()

    agent_daily_shifts['ScheduledDurationMinutes'] = (agent_daily_shifts['ScheduledShiftEnd'] - agent_daily_shifts['ScheduledShiftStart']).dt.total_seconds() / 60
    agent_daily_shifts['ActualDurationMinutes'] = (agent_daily_shifts['ActualLogout'] - agent_daily_shifts['ActualLogin']).dt.total_seconds() / 60

    agent_daily_shifts['AdherencePercentage'] = np.where(
        agent_daily_shifts['ScheduledDurationMinutes'] == 0,
        0,
        (agent_daily_shifts['ActualDurationMinutes'] / agent_daily_shifts['ScheduledDurationMinutes']) * 100
    )
    agent_daily_shifts['AdherencePercentage'] = agent_daily_shifts['AdherencePercentage'].clip(upper=100)
    return agent_daily_shifts

agent_daily_shifts = calculate_adherence(filtered_df)

@st.cache_data
def calculate_occupancy(data_frame, adherence_df):
    if data_frame.empty or adherence_df.empty:
        return pd.DataFrame(columns=adherence_df.columns.tolist() + ['ProductiveTimeMinutes', 'OccupancyPercentage'])

    productive_tasks = ['Call', 'Email', 'Chat', 'After-Call Work (ACW)']
    data_frame['IsProductive'] = data_frame['EventType'].isin(productive_tasks)
    productive_time = data_frame[data_frame['IsProductive']].groupby(['AgentID', 'Date'])['EventDurationMinutes'].sum().reset_index()
    productive_time.rename(columns={'EventDurationMinutes': 'ProductiveTimeMinutes'}, inplace=True)

    agent_performance = pd.merge(adherence_df, productive_time, on=['AgentID', 'Date'], how='left')
    agent_performance['ProductiveTimeMinutes'] = agent_performance['ProductiveTimeMinutes'].fillna(0)

    agent_performance['OccupancyPercentage'] = np.where(
        agent_performance['ActualDurationMinutes'] == 0,
        0,
        (agent_performance['ProductiveTimeMinutes'] / agent_performance['ActualDurationMinutes']) * 100
    )
    agent_performance['OccupancyPercentage'] = agent_performance['OccupancyPercentage'].replace([np.inf, -np.inf], np.nan).fillna(0)
    agent_performance['OccupancyPercentage'] = agent_performance['OccupancyPercentage'].clip(upper=100)
    return agent_performance

agent_performance = calculate_occupancy(filtered_df, agent_daily_shifts)

@st.cache_data
def calculate_shrinkage(data_frame, performance_df):
    if data_frame.empty or performance_df.empty:
        return performance_df, []

    break_types = ['Lunch', 'Short Break', 'Personal Time', 'Team Meeting', 'System Issue']
    shrinkage_types = [s for s in break_types if s not in ['Training', 'Meeting']]

    shrinkage_data = data_frame[data_frame['EventType'].isin(shrinkage_types)].groupby(['AgentID', 'Date', 'EventType'])['EventDurationMinutes'].sum().reset_index()
    shrinkage_pivot = shrinkage_data.pivot_table(index=['AgentID', 'Date'], columns='EventType', values='EventDurationMinutes', fill_value=0).reset_index()
    shrinkage_pivot.columns.name = None

    performance_df = pd.merge(performance_df, shrinkage_pivot, on=['AgentID', 'Date'], how='left').fillna(0)

    for col in shrinkage_types:
        if col not in performance_df.columns:
            performance_df[col] = 0

    performance_df['TotalShrinkageMinutes'] = performance_df[shrinkage_types].sum(axis=1)

    performance_df['ShrinkagePercentage'] = np.where(
        performance_df['ScheduledDurationMinutes'] == 0,
        0,
        (performance_df['TotalShrinkageMinutes'] / performance_df['ScheduledDurationMinutes']) * 100
    )
    performance_df['ShrinkagePercentage'] = performance_df['ShrinkagePercentage'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return performance_df, shrinkage_types

agent_performance, shrinkage_types = calculate_shrinkage(filtered_df, agent_performance)

@st.cache_data
def calculate_agent_summary(performance_df):
    if performance_df.empty:
        return pd.DataFrame(columns=['AgentID', 'AvgAdherence', 'AvgOccupancy', 'TotalProductiveTime',
                                     'TotalActualLoggedTime', 'EfficiencyScore', 'EfficiencyRank',
                                     'AdherenceOutlier', 'OccupancyOutlier', 'FlaggedOutlier'])

    agent_summary = performance_df.groupby('AgentID').agg(
        AvgAdherence=('AdherencePercentage', 'mean'),
        AvgOccupancy=('OccupancyPercentage', 'mean'),
        TotalProductiveTime=('ProductiveTimeMinutes', 'sum'),
        TotalActualLoggedTime=('ActualDurationMinutes', 'sum')
    ).reset_index()

    agent_summary['EfficiencyScore'] = agent_summary['AvgAdherence'] * agent_summary['AvgOccupancy'] / 10000
    agent_summary = agent_summary.sort_values(by='EfficiencyScore', ascending=False)
    agent_summary['EfficiencyRank'] = agent_summary['EfficiencyScore'].rank(ascending=False)

    if len(agent_summary) > 1:
        Q1_adherence = agent_summary['AvgAdherence'].quantile(0.25)
        Q3_adherence = agent_summary['AvgAdherence'].quantile(0.75)
        IQR_adherence = Q3_adherence - Q1_adherence
        agent_summary['AdherenceOutlier'] = ((agent_summary['AvgAdherence'] < (Q1_adherence - 1.5 * IQR_adherence)) |
                                           (agent_summary['AvgAdherence'] > (Q3_adherence + 1.5 * IQR_adherence)))

        Q1_occupancy = agent_summary['AvgOccupancy'].quantile(0.25)
        Q3_occupancy = agent_summary['AvgOccupancy'].quantile(0.75)
        IQR_occupancy = Q3_occupancy - Q1_occupancy
        agent_summary['OccupancyOutlier'] = ((agent_summary['AvgOccupancy'] < (Q1_occupancy - 1.5 * IQR_occupancy)) |
                                           (agent_summary['AvgOccupancy'] > (Q3_occupancy + 1.5 * IQR_occupancy)))

        agent_summary['FlaggedOutlier'] = agent_summary['AdherenceOutlier'] | agent_summary['OccupancyOutlier']
    else:
        agent_summary['AdherenceOutlier'] = False
        agent_summary['OccupancyOutlier'] = False
        agent_summary['FlaggedOutlier'] = False

    return agent_summary

agent_summary = calculate_agent_summary(agent_performance)


# --- Main Dashboard Layout ---
st.title("📊 Contact Center Workforce Optimization Dashboard")
st.markdown("---")

st.markdown("""
Welcome to the Contact Center Workforce Optimization Dashboard!
This dashboard provides key insights into agent adherence, occupancy, and shrinkage, helping you optimize operations and identify areas for improvement.
Use the filters on the sidebar to explore data for specific dates and agents.
""")

# --- Overall Team Performance ---
st.header("📈 Overall Team Performance")
st.markdown("---")

if agent_performance.empty:
    st.info("No agent performance data available for the selected filters to calculate overall team metrics.")
    avg_adherence_val = 0.0
    avg_occupancy_val = 0.0
    total_productive_time_hrs_val = 0.0
    total_logged_in_time_hrs_val = 0.0
else:
    avg_adherence_val = agent_performance['AdherencePercentage'].mean()
    avg_occupancy_val = agent_performance['OccupancyPercentage'].mean()
    total_productive_time_hrs_val = agent_performance['ProductiveTimeMinutes'].sum() / 60
    total_logged_in_time_hrs_val = agent_performance['ActualDurationMinutes'].sum() / 60

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Average Adherence", value=f"{avg_adherence_val:.2f}%")
with col2:
    st.metric(label="Average Occupancy", value=f"{avg_occupancy_val:.2f}%")
with col3:
    st.metric(label="Total Productive Time", value=f"{total_productive_time_hrs_val:.2f} Hrs")
with col4:
    st.metric(label="Total Logged-in Time", value=f"{total_logged_in_time_hrs_val:.2f} Hrs")

overall_team_summary = pd.DataFrame({
    'Metric': ['Average Adherence (%)', 'Average Occupancy (%)', 'Total Productive Time (Hrs)', 'Total Logged-in Time (Hrs)'],
    'Value': [avg_adherence_val, avg_occupancy_val, total_productive_time_hrs_val, total_logged_in_time_hrs_val]
}).set_index('Metric')

with st.expander("View Overall Team Performance Table"):
    st.dataframe(overall_team_summary.round(2))

st.markdown("---")


# --- Performance Trend Analysis ---
st.header("📊 Performance Trend Analysis")
st.markdown("---")

if not agent_performance.empty:
    st.subheader("Daily Adherence Trends")
    team_daily_adherence = agent_performance.groupby('Date')['AdherencePercentage'].mean().reset_index()
    fig_daily_adherence = px.line(team_daily_adherence, x='Date', y='AdherencePercentage',
                                  title='Team Average Daily Adherence',
                                  labels={'AdherencePercentage': 'Adherence (%)'},
                                  template='plotly_white')
    st.plotly_chart(fig_daily_adherence, use_container_width=True)

    st.subheader("Weekly Adherence Trends")
    temp_df_for_weekly = agent_performance.copy()
    temp_df_for_weekly['Date_dt'] = pd.to_datetime(temp_df_for_weekly['Date'])
    temp_df_for_weekly['Week'] = temp_df_for_weekly['Date_dt'].dt.isocalendar().week.astype(int)
    temp_df_for_weekly['Year'] = temp_df_for_weekly['Date_dt'].dt.year
    team_weekly_adherence = temp_df_for_weekly.groupby(['Year', 'Week'])['AdherencePercentage'].mean().reset_index()
    team_weekly_adherence['Week_Label'] = team_weekly_adherence['Year'].astype(str) + '-W' + team_weekly_adherence['Week'].astype(str).str.zfill(2)

    fig_weekly_adherence = px.bar(team_weekly_adherence, x='Week_Label', y='AdherencePercentage',
                                  title='Team Average Weekly Adherence',
                                  labels={'AdherencePercentage': 'Adherence (%)', 'Week_Label': 'Year-Week'},
                                  template='plotly_white')
    st.plotly_chart(fig_weekly_adherence, use_container_width=True)


    st.subheader("Daily Occupancy Trends")
    team_daily_occupancy = agent_performance.groupby('Date')['OccupancyPercentage'].mean().reset_index()
    fig_daily_occupancy = px.line(team_daily_occupancy, x='Date', y='OccupancyPercentage',
                                  title='Team Average Daily Occupancy',
                                  labels={'OccupancyPercentage': 'Occupancy (%)'},
                                  template='plotly_white',
                                  color_discrete_sequence=['red'])
    st.plotly_chart(fig_daily_occupancy, use_container_width=True)


    st.subheader("Daily Shrinkage Percentage Trends")
    team_daily_shrinkage = agent_performance.groupby('Date')['ShrinkagePercentage'].mean().reset_index()
    fig_daily_shrinkage = px.line(team_daily_shrinkage, x='Date', y='ShrinkagePercentage',
                                  title='Team Average Daily Shrinkage Percentage',
                                  labels={'ShrinkagePercentage': 'Shrinkage (%)'},
                                  template='plotly_white',
                                  color_discrete_sequence=['purple'])
    st.plotly_chart(fig_daily_shrinkage, use_container_width=True)
else:
    st.info("No data available for trend analysis with the current filters.")


st.markdown("---")

# --- Agent-Level Performance ---
st.header("👤 Agent Performance Insights")
st.markdown("---")

if not agent_summary.empty:
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Distribution of Average Adherence")
        fig_adherence_dist = px.histogram(agent_summary, x='AvgAdherence', nbins=20,
                                         title='Distribution of Agent Average Adherence',
                                         labels={'AvgAdherence': 'Average Adherence (%)'},
                                         template='plotly_white')
        st.plotly_chart(fig_adherence_dist, use_container_width=True)

    with col6:
        st.subheader("Distribution of Average Occupancy")
        fig_occupancy_dist = px.histogram(agent_summary, x='AvgOccupancy', nbins=20,
                                         title='Distribution of Agent Average Occupancy',
                                         labels={'AvgOccupancy': 'Average Occupancy (%)'},
                                         template='plotly_white',
                                         color_discrete_sequence=['red'])
        st.plotly_chart(fig_occupancy_dist, use_container_width=True)

    st.subheader("Agent Adherence vs. Occupancy (with Outliers)")
    fig_scatter = px.scatter(agent_summary, x='AvgAdherence', y='AvgOccupancy',
                             color='FlaggedOutlier',
                             hover_data=['AgentID', 'EfficiencyScore'],
                             title='Agent Adherence vs. Occupancy',
                             labels={'AvgAdherence': 'Average Adherence (%)', 'AvgOccupancy': 'Average Occupancy (%)',
                                     'FlaggedOutlier': 'Flagged Outlier'},
                             template='plotly_white')
    fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig_scatter, use_container_width=True)


    st.subheader("Top & Bottom Agents by Efficiency")
    col7, col8 = st.columns(2)

    with col7:
        st.markdown("##### Top 5 Agents")
        top_5_agents = agent_summary.head(5)[['AgentID', 'EfficiencyScore', 'AvgAdherence', 'AvgOccupancy']].round(2)
        st.dataframe(top_5_agents, hide_index=True)

    with col8:
        st.markdown("##### Bottom 5 Agents")
        bottom_5_agents = agent_summary.tail(5)[['AgentID', 'EfficiencyScore', 'AvgAdherence', 'AvgOccupancy']].round(2)
        st.dataframe(bottom_5_agents, hide_index=True)

    st.subheader("Agent Efficiency Ranking")
    if len(agent_summary) > 30:
        top_bottom_agents_for_chart = pd.concat([agent_summary.head(15), agent_summary.tail(15)]).sort_values(by='EfficiencyScore', ascending=True)
    else:
        top_bottom_agents_for_chart = agent_summary.sort_values(by='EfficiencyScore', ascending=True)

    fig_efficiency_rank = px.bar(top_bottom_agents_for_chart, x='EfficiencyScore', y='AgentID', orientation='h',
                                 title='Agent Efficiency Score Ranking (Top & Bottom Agents)',
                                 labels={'EfficiencyScore': 'Efficiency Score', 'AgentID': 'Agent ID'},
                                 template='plotly_white',
                                 color='EfficiencyScore', color_continuous_scale=px.colors.sequential.Greens)
    fig_efficiency_rank.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_efficiency_rank, use_container_width=True)
else:
    st.info("No agent-level data available for detailed insights with the current filters.")

st.markdown("---")

# --- Shrinkage Breakdown ---
st.header("🕰️ Shrinkage Analysis")
st.markdown("---")

total_shrinkage_by_type_df = pd.DataFrame()
if not filtered_df.empty and shrinkage_types:
    total_shrinkage_by_type_df = filtered_df[filtered_df['EventType'].isin(shrinkage_types)].groupby('EventType')['EventDurationMinutes'].sum().reset_index()
    total_shrinkage_by_type_df.rename(columns={'EventDurationMinutes': 'TotalMinutes'}, inplace=True)
    total_shrinkage_by_type_df['TotalHours'] = total_shrinkage_by_type_df['TotalMinutes'] / 60
    total_shrinkage_by_type_df = total_shrinkage_by_type_df.sort_values(by='TotalMinutes', ascending=False).round(2)

if not total_shrinkage_by_type_df.empty:
    st.subheader("Total Shrinkage Breakdown by Category")
    col9, col10 = st.columns([1, 2])

    with col9:
        st.markdown("##### Detailed Breakdown")
        st.dataframe(total_shrinkage_by_type_df, hide_index=True)

    with col10:
        st.markdown("##### Visual Overview")
        fig_shrinkage_breakdown = px.bar(total_shrinkage_by_type_df, x='EventType', y='TotalHours',
                                         title='Total Shrinkage Breakdown by Category',
                                         labels={'EventType': 'Shrinkage Type', 'TotalHours': 'Total Hours'},
                                         template='plotly_white',
                                         color='TotalHours', color_continuous_scale=px.colors.sequential.Viridis)
        fig_shrinkage_breakdown.update_layout(xaxis_title="Shrinkage Category", yaxis_title="Total Hours")
        st.plotly_chart(fig_shrinkage_breakdown, use_container_width=True)
else:
    st.info("No shrinkage data available for analysis with the current filters.")

st.markdown("---")


# --- Team-Level Recommendations ---
st.header("💡 Team-Level Recommendations")
st.markdown("---")

avg_adherence = avg_adherence_val
avg_occupancy = avg_occupancy_val
total_flagged_outliers = agent_summary['FlaggedOutlier'].sum()

top_shrinkage_type = total_shrinkage_by_type_df.iloc[0]['EventType'] if not total_shrinkage_by_type_df.empty else "N/A"
top_shrinkage_hours = total_shrinkage_by_type_df.iloc[0]['TotalHours'] if not total_shrinkage_by_type_df.empty else 0

st.markdown(f"Based on the analysis of contact center compliance data for the selected period:")

st.markdown("#### 1. Overall Performance")
st.markdown(f"- The team's **average adherence** is **{avg_adherence:.2f}%**, indicating agents are generally logging in and out close to their scheduled times.")
st.markdown(f"- The **average occupancy** is **{avg_occupancy:.2f}%**, reflecting the proportion of logged-in time spent on productive tasks.")

st.markdown("#### 2. Adherence and Occupancy Insights")
if avg_adherence < 90:
    st.warning(f"**Adherence Focus Needed:** The average adherence is **below 90%**. Investigate reasons for discrepancies between scheduled and actual login/logout times. This could be due to system issues, late starts, early finishes, or unrecorded breaks. Review login/logout procedures and provide refresher training if needed.")
else:
    st.success(f"**Adherence Strength:** Adherence is strong at {avg_adherence:.2f}%. Continue to monitor to ensure consistency.")

if avg_occupancy < 70:
    st.warning(f"**Occupancy Improvement:** The average occupancy is **below 70%**. This suggests there might be significant idle time or time spent on non-productive activities while logged in. Analyze detailed agent activity logs for patterns in non-productive time. Optimize workflow, reduce unnecessary tasks, or re-evaluate staffing levels.")
elif avg_occupancy > 85:
    st.info(f"**High Occupancy Alert:** Occupancy is quite high at {avg_occupancy:.2f}%, which could indicate potential for agent burnout or insufficient break times. Ensure agents are taking their scheduled breaks and have adequate time for after-call work. Consider staffing adjustments if consistently high.")
else:
    st.success(f"**Occupancy Stability:** Occupancy is within a healthy range ({avg_occupancy:.2f}%). Continue to balance productivity with agent well-being.")

st.markdown("#### 3. Shrinkage Patterns")
if top_shrinkage_hours > 0 and top_shrinkage_type != "N/A":
    st.info(f"The most significant shrinkage category identified is **'{top_shrinkage_type}'**, accounting for **{top_shrinkage_hours:.2f} hours** during the period. Analyze the root causes for high shrinkage in this category. For example, if 'Personal Time' is high, review policies or provide clearer guidelines. If 'System Issue' is high, escalate to IT for resolution.")
else:
    st.success("No significant shrinkage patterns were identified, which is positive.")

st.markdown("#### 4. Agent Performance & Outliers")
st.markdown(f"The **efficiency score** provides a combined view of adherence and occupancy. Regularly review the ranking to identify top performers for recognition and agents who may need support.")
if total_flagged_outliers > 0:
    st.warning(f"**Outlier Action:** **{int(total_flagged_outliers)} agent(s)** were flagged as outliers in either adherence or occupancy. Conduct individual coaching sessions with these agents to understand challenges and provide targeted support or training. Investigate if there are systemic issues contributing to their outlier status.")
else:
    st.success("No significant outliers were detected, indicating generally consistent team performance.")

st.markdown("#### 5. Planning Effectiveness & KPI Tracking")
st.markdown("- Use adherence and shrinkage data to **refine future staffing forecasts and schedules**. Accurate scheduling directly impacts efficiency and customer service levels.")
st.markdown("- Continuously **monitor adherence, occupancy, and shrinkage as key performance indicators (KPIs)**. Set realistic targets and track progress over time.")
st.markdown("- Establish a **feedback loop** where insights from this monitoring tool are regularly communicated to team leads and agents to drive continuous improvement.")
st.markdown("- Identify and address any operational bottlenecks or process inefficiencies highlighted by low occupancy or high shrinkage in specific task types.")

st.markdown("---")
st.success("Analysis Complete!")
