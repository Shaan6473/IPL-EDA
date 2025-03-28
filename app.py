import streamlit as st
import homePage
import exploratoryDataAnalysis
import ipl_GenAI
#import datasetPreprocessing
#import scrollToTop


st.set_page_config(
    page_title="IPL ANALYSIS",
    page_icon="🏏",
    initial_sidebar_state='expanded',
    layout="wide"
)


st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """,
    unsafe_allow_html=True,
)

PAGES = {
    "HOME": homePage,
    "Exploratory Data Analysis": exploratoryDataAnalysis,
    "Query Bot": ipl_GenAI
}


st.sidebar.title('NAVIGATION')
selection = st.sidebar.radio('', list(PAGES.keys()))
page = PAGES[selection]
page.app()
