import streamlit as st


def main(path):
    st.subheader("Power BI report:")
    st.markdown(f'<iframe width="900" height="541.25" src="{path}" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True)


report = ("https://app.powerbi.com/reportEmbed?reportId=b99a36f5-13a5-44fc-8846-76919dbef12d&autoAuth=true&ctid="
          "9ea332bc-d5e1-4211-a36b-1e06be85ea93")

main(path=report)
