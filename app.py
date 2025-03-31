import streamlit as st
import pandas as pd
import numpy as np


from oneway.test_first import test_agent as modelone
from oneway.test_second import test_agent as modelone_second
from oneway.ttl_first import run_simulation as ttlone
from oneway.ttl_second import run_simulation as ttlone_second
from twoway.test import test_agent as modeltwo
from twoway.ttl import run_simulation as ttltwo

st.set_page_config(page_title="DRATS", layout="wide")
st.title("Deep Reinforcement Approach on Traffic System Dashboard")

# Sidebar for simulation selection
intersection_choice = st.sidebar.radio("Select Intersections", ("One Intersection", "Two Intersections"))
st.sidebar.write("You selected:", intersection_choice)
if intersection_choice == "One Intersection":
    st.header("One Fourway Intersection")
    st.subheader("Training Rewards")
    st.image("oneway/images/Training_Rewards.jpg")
    sit_choice = st.radio("Select type of situation:", ("High traffic", "Low traffic", "Flow-controlled traffic"))
    if sit_choice == "High traffic":
        st.write("The model is trained to handle high traffic from all the intersections efficiently.")

    if sit_choice == "Low traffic":
        st.write("Here we will have scarcely any traffic.")

    if sit_choice == "Flow-controlled traffic":
        st.write("you can select from which directions the traffic flow should occur.")

        checkbox_values = [0, 0, 0, 0]

        # Create columns for horizontal layout
        cols = st.columns(4)

        # Add checkboxes in each column
        with cols[0]:
            if st.checkbox("North"):
                checkbox_values[0] = 1
        with cols[1]:
            if st.checkbox("East"):
                checkbox_values[1] = 1
        with cols[2]:
            if st.checkbox("South"):
                checkbox_values[2] = 1
        with cols[3]:
            if st.checkbox("West"):
                checkbox_values[3] = 1
    if st.button("Test"):
        if sit_choice == "High traffic":
            rewards_agent, waiting_times_agent = modelone()
            rewards_ttl, waiting_times_ttl = ttlone()
        if sit_choice == "Low traffic":
            rewards_agent, waiting_times_agent = modelone_second(sumo_cmd = ["sumo-gui", "-c", "oneway/one_intersection/second.sumocfg"])
            rewards_ttl, waiting_times_ttl = ttlone_second(sumo_cmd = ["sumo-gui", "-c", "oneway/one_intersection/second.sumocfg"])
        if sit_choice == "Flow-controlled traffic":
            if checkbox_values != [0, 0, 0, 0]:
                rewards_agent, waiting_times_agent = modelone_second(choices=checkbox_values, sumo_cmd = ["sumo-gui", "-c", "oneway/one_intersection/third.sumocfg"])
                rewards_ttl, waiting_times_ttl = ttlone_second(sumo_cmd = ["sumo-gui", "-c", "oneway/one_intersection/third.sumocfg"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rewards over timesteps during testing")
            st.line_chart(pd.DataFrame({"Rewards over timesteps by RL agent": rewards_agent}))
        with col2:
            st.subheader("Waiting times over timesteps during testing")
            st.line_chart(pd.DataFrame({"Waiting times over timesteps by RL agent": waiting_times_agent}))
            st.write("Average waiting time by RL agent = ", np.mean(waiting_times_agent))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rewards over timesteps from TTLS")
            st.line_chart(pd.DataFrame({"Rewards over timesteps from TTLS": rewards_ttl}))
        with col2:
            st.subheader("Waiting times over timesteps from TTLS")
            st.line_chart(pd.DataFrame({"Waiting times over timesteps from TTLS": waiting_times_ttl}))
            st.write("Average waiting time by TTLS = ", np.mean(waiting_times_ttl))

if intersection_choice == "Two Intersections":
    st.header("Two Fourway Intersections")
    st.subheader("Training Rewards")
    st.image("twoway/images/Training Rewards.jpg")
    if st.button("Test"):
        # subprocess.run(["python", "test_first.py"])
        rewards, waiting_times = modeltwo()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rewards over timesteps during testing")
            st.line_chart(pd.DataFrame({"Rewards over timesteps during testing": rewards}))
        with col2:
            st.subheader("Waiting times over timesteps during testing")
            st.line_chart(pd.DataFrame({"Waiting times over timesteps during testing": waiting_times}))

        rewards, waiting_times = ttltwo()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rewards over timesteps from TTLS")
            st.line_chart(pd.DataFrame({"Rewards over timesteps from TTLS": rewards}))
        with col2:
            st.subheader("Waiting times over timesteps from TTLS")
            st.line_chart(pd.DataFrame({"Waiting times over timesteps from TTLS": waiting_times}))

