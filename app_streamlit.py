import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import base64

from src.pipeline.prediction_pipeline import PredictPipeline
from src.constants.config import *


backgroundColor = st.get_option("theme.backgroundColor")
black_background = f"<style>:root {{background-color: {backgroundColor};}}</style>"


def add_extra_space(times: int):
    for _ in range(times):
        st.write('\n')


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


def main():
    set_png_as_page_bg('dark.jpg')
    st.title("GNN-powered Classification and Explanation App")
    add_extra_space(6)
    option = st.selectbox(
        'Select the operation you wanna perform:',
        ('node classification through Cora Dataset',
         'graph classification through ENZYMES Dataset'))
    # placeholder = st.empty()
    if option == 'node classification through Cora Dataset':
        input_chosen = st.number_input(
            "Type Node ID to classify and explain",
            min_value=0, max_value=2707,
            value=-1)
        val_chosen = int(input_chosen)
        if val_chosen >= 0:
            # placeholder.number_input("Type Node ID to classify and explain",
            #                          min_value=0, max_value=2707, key='1',
            #                          value=val_chosen, disabled=True)
            with st.spinner("Executing the request..."):
                # print("in condition")
                obj = PredictPipeline(val_chosen)
                obj.load_model()
                predicted_class = obj.predict()
                explainer, top_k = obj.train_and_explain()
                data = obj.prepare_feature_mask(explainer, top_k)
                bar_figure = px.bar(data, x="score", y="feature_id", orientation='h',
                                    hover_data=["score", "feature_id"],
                                    height=400,
                                    title=f"Feature importance for top {len(data)} features")
                bar_figure.update_layout(
                    hoverlabel=dict(
                        bgcolor="#828282",
                        bordercolor="white"
                    )
                )
                tab1, tab2, tab3, tab4 = st.tabs([
                    'Predicted Class', 'Original Graph',
                    'Feature Importance', 'Explained Graph'
                ])
                # obj.visualize_node_subgraph()

                with tab1:
                    st.success(f"Predicted class is: '{predicted_class}'")
                with tab2:
                    st.header("Subgraph for the selected Node ID")
                    # backgroundColor = st.get_option("theme.backgroundColor")
                    # black_background = f"<style>:root {{background-color: {backgroundColor};}}</style>"
                    st.info("Green-colored node reflects the node with the selected Node ID", icon="ℹ️")
                    obj.visualize_node_subgraph()
                    html_file = open(GraphPath.node_subgraph.value, 'r', encoding='utf-8')
                    source_code = html_file.read()
                    components.html(source_code + black_background, height=610)
                    # st.text("Green-colored node reflects the node with the selected Node ID")
                with tab3:
                    st.plotly_chart(bar_figure, True)
                with tab4:
                    # st.header("Explanation Graph for the selected Graph ID")
                    # st.info("Important edges (part of learned edge mask, if any) have been "
                    #         "displayed in 'Orange' color", icon="ℹ️")
                    # backgroundColor = st.get_option("theme.backgroundColor")
                    # black_background = f"<style>:root {{background-color: {backgroundColor};}}</style>"
                    # # st.cache(allow_output_mutation=True)
                    explanation_check = obj.visualize_explanation_subgraph(explainer)
                    # print(explanation_check)
                    # html_file = open(GraphPath.node_net_graph.value, 'r', encoding='utf-8')
                    # source_code = html_file.read()
                    # components.html(source_code + black_background, height=790)
                    if explanation_check:
                        # asyncio.run(display_graph())
                        # loop = asyncio.get_event_loop()
                        # loop.run_until_complete(display_graph())
                        st.header("Extracted Explanation Subgraph for the selected Node ID")
                        small_graph_html_file = open(GraphPath.node_extract_graph.value, 'r', encoding='utf-8')
                        small_graph_source_code = small_graph_html_file.read()
                        components.html(small_graph_source_code + black_background, height=610)
            # placeholder.number_input("Type Node ID to classify and explain",
            #                          min_value=0, max_value=2707, key='2',
            #                          value=-1, disabled=False)
        else:
            st.info("Please select Node ID between 0 and 2707 (inclusive) to make predictions...", icon="ℹ️")
    else:
        # graph_placeholder = st.empty()
        input_chosen = st.number_input(
            "Type Graph ID to classify and explain",
            min_value=0, max_value=599, key='value1',
            value=-1)
        val_chosen = int(input_chosen)
        # print(f'val_chosen: {val_chosen, st.session_state.value1}')
        if val_chosen >= 0:
            # graph_placeholder.number_input(
            #     "Type Graph ID to classify and explain",
            #     min_value=0, max_value=599, key='value2',
            #     value=st.session_state.value1, disabled=True)
            with st.spinner("Executing the request..."):
                # print("in condition")
                obj = PredictPipeline(val_chosen, task='graph')
                obj.load_model()
                predicted_class = obj.predict()
                explainer, top_k = obj.train_and_explain()
                data = obj.prepare_feature_mask(explainer, top_k)
                bar_figure = px.bar(data, x="score", y="feature_id", orientation='h',
                                    hover_data=["score", "feature_id"],
                                    height=400,
                                    title=f"Feature importance for all {len(data)} features")
                bar_figure.update_layout(
                    hoverlabel=dict(
                        bgcolor="#828282",
                        bordercolor="white"
                    )
                )
                tab1, tab2, tab3, tab4 = st.tabs([
                    'Predicted Class', 'Original Subgraph',
                    'Feature Importance', 'Explained Subgraph'
                ])
                # obj.visualize_node_subgraph()

                with tab1:
                    st.success(f"Predicted class is: '{predicted_class}'")
                with tab2:
                    st.header("Graph for the selected Graph ID")
                    # backgroundColor = st.get_option("theme.backgroundColor")
                    # black_background = f"<style>:root {{background-color: {backgroundColor};}}</style>"
                    obj.visualize_original_graph()
                    html_file = open(GraphPath.original_graph.value, 'r', encoding='utf-8')
                    source_code = html_file.read()
                    components.html(source_code + black_background, height=610)
                with tab3:
                    st.plotly_chart(bar_figure, True)
                with tab4:
                    st.header("Explanation Graph for the selected Graph ID")
                    # backgroundColor = st.get_option("theme.backgroundColor")
                    # black_background = f"<style>:root {{background-color: {backgroundColor};}}</style>"
                    st.info("Important edges (part of learned edge mask, if any) have been "
                            "displayed in 'Orange' color", icon="ℹ️")
                    # st.cache(allow_output_mutation=True)
                    explanation_check = obj.visualize_explanation_subgraph(explainer)
                    html_file = open(GraphPath.graph_net_graph.value, 'r', encoding='utf-8')
                    source_code = html_file.read()
                    components.html(source_code + black_background, height=790)
                    # if explanation_check:
                    #     st.header("Extracted Explanation Subgraph")
                    #     small_graph_html_file = open(GraphPath.graph_extract_graph.value, 'r', encoding='utf-8')
                    #     small_graph_source_code = small_graph_html_file.read()
                    #     components.html(small_graph_source_code + black_background, height=610)
            # graph_placeholder.number_input(
            #     "Type Graph ID to classify and explain",
            #     min_value=0, max_value=599, key='value3',
            #     value=-1, disabled=False)
        else:
            st.info("Please select Graph ID between 0 and 599 (inclusive) to make predictions...", icon="ℹ️")


if __name__ == '__main__':
    main()
