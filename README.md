# Contact Center Workforce Optimization Dashboard

## Created By
###Andiswa Mabuza
###LinkedIn: https://linkedin.com/in/andiswamabuza
###email: amabuza53@gmail.com

## Overview

Welcome to the **Contact Center Workforce Optimization Dashboard**! This project provides a powerful, data-driven solution for analyzing and optimizing the performance of contact center agents. It leverages a dataset **I created and exported from a Kaggle environment**, focusing on key metrics such as **adherence**, **occupancy**, and **shrinkage**. The dashboard helps managers gain deep insights into operational efficiency, identify areas for improvement, and make informed decisions for **strategic resource planning**.

The application performs robust data processing, offers interactive visualizations, and provides actionable recommendations to empower contact center leadership. Built with Python, Pandas, Plotly, and Streamlit, it delivers an intuitive and professional user experience.

**✨ Live Demo:** You can explore the dashboard live here: [https://contact-center-workforce-optimization-dashboard.streamlit.app](https://contact-center-workforce-optimization-dashboard.streamlit.app)

---

## Features

* **Custom Dataset Integration**: Utilizes a pre-prepared synthetic yet realistic dataset that I created and exported from a Kaggle environment, ensuring relevant and comprehensive data for analysis.
* **Adherence Tracking**: Calculates and visualizes how well agents stick to their scheduled shifts.
* **Occupancy Monitoring**: Measures the percentage of time agents spend on productive tasks while logged in.
* **Shrinkage Analysis**: Breaks down non-productive time into categories (e.g., breaks, system issues) to identify areas for reduction.
* **Agent Efficiency Scoring**: Ranks agents based on a composite efficiency score derived from adherence and occupancy.
* **Outlier Detection**: Automatically flags agents who fall outside typical performance ranges in adherence or occupancy.
* **Interactive Visualizations**: Uses Plotly Express to provide dynamic and engaging charts for trends and distributions.
* **Detailed Performance Summaries**: Presents key performance indicators (KPIs) in clear, concise tables.
* **Actionable Recommendations**: Offers narrative insights and practical advice based on the aggregated data.
* **Streamlit Web Application**: Provides an easy-to-use, interactive web interface with date range and agent filtering.

## Project Structure

.
├── app.py                     # Main Streamlit application
├── agent_performance_data.csv # Dataset (created in Kaggle and placed here for the app deployment)
└── README.md                  # This ReadMe file

## Getting Started

To experience the dashboard, simply visit the **Live Demo** link provided above. The application is hosted on Streamlit Cloud, making it immediately accessible without any local setup.

### Data Source

This dashboard uses a dataset that was created by you in a Kaggle environment and then exported as a CSV. This `agent_performance_data.csv` file is included in the project's deployment.

## Usage

Once you're on the live application:

* **Sidebar Filters**: Use the sidebar on the left to select a **date range** and **specific agents** to analyze. The dashboard will dynamically update based on your selections.
* **Dashboard Sections**: Explore different aspects of performance:
    * **Overall Team Performance**: Key metrics at a glance.
    * **Performance Trend Analysis**: Daily and weekly trends for adherence, occupancy, and shrinkage.
    * **Agent Performance Insights**: Distributions of adherence and occupancy, and a scatter plot highlighting outliers.
    * **Shrinkage Analysis**: Breakdown of non-productive time categories.
    * **Team-Level Recommendations**: Actionable insights based on the aggregated data.

## Technical Details

* **Python Libraries**:
    * `pandas`: For data manipulation and analysis.
    * `numpy`: For numerical operations.
    * `streamlit`: For building the interactive web application.
    * `plotly.express`: For creating rich, interactive, and visually appealing charts.
* **Data Structure**: The application processes data reflecting individual agent activities, scheduled shifts, actual login/logout times, and various event types (productive tasks, breaks, etc.).
* **Key Calculations**:
    * **Adherence**: `(Actual Logged-in Time / Scheduled Shift Time) * 100%`
    * **Occupancy**: `(Productive Time / Actual Logged-in Time) * 100%`
    * **Efficiency Score**: `(Avg Adherence * Avg Occupancy) / 10000` (normalized)
    * **Outlier Detection**: Uses the Interquartile Range (IQR) method to identify agents with unusually high or low adherence/occupancy.
* **Caching**: Streamlit's `@st.cache_data` decorator is used extensively to optimize performance by caching costly data loading and processing functions.
