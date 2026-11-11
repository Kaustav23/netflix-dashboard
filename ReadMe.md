# Netflix Catalog EDA & Streamlit Dashboard

An end-to-end Exploratory Data Analysis (EDA) and interactive web dashboard built to analyze the Netflix streaming catalog. This project processes a dataset of over 8,700 Netflix movies and TV shows to extract business KPIs, production trends, global footprint maps, and talent network clusters.

## Features

* **Phase 1: Data Preprocessing & Cleaning:** Handles date parsing, missing value imputation ("Unknown"), and regex-based multi-value exploding to accurately count collaborative international projects and ensemble casts.
* **Phase 2: Core Business KPIs:** Live metrics tracking Total Catalog Volume, Format Mix (Movies vs. TV Shows), Top Production Hubs, and 3-Year Acquisition Velocity.
* **Phase 3: Interactive Visualizations:**
  * **Acquisition Trends:** Time-series analysis answering whether the platform is shifting focus from feature films to serialized television.
  * **Global Footprint:** Choropleth maps and horizontal bar charts visualizing the top 15 content-producing nations.
  * **Production Network Analysis:** Scatter plots and network graphs identifying recurring collaborations between prolific actors and directors (e.g., specific animation studios and Bollywood clusters).
  * **Thematic Dominance:** Genre distribution charts and Natural Language Processing (NLP) word clouds extracting core narrative themes from title synopses.
* **Phase 4: Strategic Insights:** Automated summaries covering content localization strategies, the structural evolution of viewing formats, and maturity rating distributions.

## Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization & Analytics:** Plotly, Matplotlib, NetworkX, WordCloud

## Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Kaustav23/netflix-dashboard.git](https://github.com/Kaustav23/netflix-dashboard.git)
   cd netflix-dashboard