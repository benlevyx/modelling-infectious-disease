"""Main module for the streamlit app"""
import streamlit as st
import flu_eda_streamlit as eda
import st_data_preprocessing as data_preproc
import st_homepage as hp
import st_flu_inference as inference
import st_flu_forecasting as flu_forcast
import covid_forecasting as covid
import st_conclusion as concl


def main():
    page = st.sidebar.selectbox("Choose a page", ["Homepage",
                                                  "EDA",
                                                  "Data Preprocessing",
                                                  "Flu Inference",
                                                  "Flu Forecasting",
                                                  "COVID-19 Forecasting",
                                                  "Conclusion"])
    if page == "Homepage":
        hp.main()
    elif page == "EDA":
        eda.main()
    elif page == "Data Preprocessing":
        data_preproc.main()
    elif page == "Flu Inference":
        inference.main()
    elif page == "Flu Forecasting":
        flu_forcast.main()
    if page == "COVID-19 Forecasting":
        covid.main()
    elif page == "Conclusion":
        concl.main()


#add about section in the end
if __name__ == "__main__":
    main()