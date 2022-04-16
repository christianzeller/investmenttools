import streamlit as st
import supertrend_analyzer as sta
import portfolio_analyzer as pa

sa = sta.analyzer(st)
pa = pa.analyzer(st)
