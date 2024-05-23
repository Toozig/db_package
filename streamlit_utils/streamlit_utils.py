import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.errors import StreamlitAPIException
import seaborn as sns
import os
import json
import tempfile
import streamlit.components.v1 as components
import pages.db_package.process_data.process_data_utils.pbm_data_prep as pbu


FLOW_CHART_PATH = '/home/dsi/toozig/gonen-lab/users/toozig/projects/deepBind_pipeline/streamlit_app/pages/db_package/streamlit_utils/model_flow_chart.json'

def check_bed_file(bed_file, title):
    st.write(title)
    col1, col2 = st.columns(2)
    bed = load_and_check_bed_file(bed_file)
    if bed is None:
        return False

    with col1:
        plot_interval_lengths(bed)

    with col2:
        plot_segments_per_chromosome(bed)

    return True

def load_and_check_bed_file(bed_file):
    try :
        bed = pd.read_csv(bed_file, sep='\t', header='infer')
        if bed.empty:
            st.write("The BED file has no entries.")
            return None
        return bed
    except Exception as e:
        st.write(f"An error occurred: {e}")
        raise StreamlitAPIException(f'Stopping app due to error with BED file - {bed_file}')
    

def plot_interval_lengths(bed):
    # get the names of colums 2 and 1
    col1 = bed.columns[1]
    col2 = bed.columns[2]
    sns.histplot( bed[col2] - bed[col1] , bins=50)
    # print( bed[2] - bed[1] )
    plt.title("Histogram of interval lengths")
    plt.xlabel("Length")
    plt.ylabel("Count")
    st.pyplot(plt)
    plt.close()

def plot_segments_per_chromosome(bed):
    chrom_counts = bed[bed.columns[0]].value_counts()
    sns.barplot(x=chrom_counts.index, y=chrom_counts.values)
    plt.title("Number of segments per chromosome")
    # rotate x label 90
    plt.xticks(rotation=90)
    plt.xlabel("Chromosome")
    plt.ylabel("Count")
    st.pyplot(plt)
    plt.close()

def have_cols(path, columns_to_check, table_name, header=None):
    if not len(path):
        return True
    try:
        df = pbu.open_file(path, header)
    except:
        st.error(f'error opening the file {path}')
    missing_columns = [col for col in columns_to_check if col not in df.columns and len(col)]
    if missing_columns:
        st.error(f"The following columns are missing in the {table_name} table: {', '.join(missing_columns)}")
    return False # no missing columns!





def plot_histogram(df, number_col, seq_col):
    # Plot the histogram using seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(df[number_col], bins=30, kde=True)
    plt.title(f'Histogram of PBM score (n samples: {len(df)})')
    plt.xlabel('pbm score')
    plt.ylabel('Frequency')

    first_item_length = len(df[seq_col].values[0])
    header_text = f' Length of first item in {seq_col}: {first_item_length}'
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    st.write(header_text)


def get_file_from_user(name):
    path = st.text_input(label=f'Enter {name}_path')
    st.write('Or')
    uploaded_file = st.file_uploader("Choose a file", key=name)
    if uploaded_file is not None:
        # Save the file with its original name in a temporary directory
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, 'wb') as f:
            f.write(uploaded_file.getvalue())
    return path




def exp_details(name):
    path = get_file_from_user(name)
    cite = st.text_input(label=f'Enter cite_{name}')
    return path , cite

def open_file(path, header):
    try:
        return pbu.open_file(path, header)
    except FileNotFoundError:
            st.error('File not found. Please check the path and try again.')


def remove_linker(df, linker, seq_col):
    if len(linker):
        df[seq_col] = df[seq_col].str.replace(linker, '')
    return df

def show_table(path, header, linker, seq_col, placeholder):
    df = open_file(path, header)

    df = remove_linker(df, linker, seq_col)
    with placeholder.container():
        st.write(f'{path} sample')
        st.write (f'columns names: {" ".join(df.columns.tolist())}')
        if linker:
            st.write('The linker will be removed from the sequence')
        st.write(df.head(4))
    return df




def show_single_histogram(path, signal_col, background_col, seq_col,to_norm ,header, linker):
    df = open_file(path, header)
    df = remove_linker(df, linker, seq_col)
    st.write(df.head())

    values = pbu.normalize_PBM_target(df, signal_col, background_col) if to_norm else df[signal_col].to_numpy()
    df['to_plot'] = values
    plot_histogram(df, 'to_plot', seq_col)
        


def show_histograms(path_ME, path_HK, path, HKME, signal_col, background_col, seq_col,to_norm,header, linker):
    have_file = len(path) or len(path_ME) or len(path_HK) 

    if bool(have_file):
        if HKME:
            show_single_histogram(path_ME, signal_col, background_col, seq_col, to_norm, header, linker)
            show_single_histogram(path_HK, signal_col, background_col, seq_col, to_norm, header, linker)
        else:
            show_single_histogram(path, signal_col, background_col, seq_col, to_norm, header, linker)


def head(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[:4]
            st.code('\n'.join(lines))
    except FileNotFoundError:
        st.error('File not found. Please check the path and try again.')



def show_file_samples(path_ME, path_HK, path, HKME, header, place1,place2, linker,seq_col):
    have_file = len(path) or len(path_ME) or len(path_HK)
    if have_file:
        if HKME:
            show_table(path_ME, header, linker, seq_col, place1)
            show_table(path_HK, header, linker, seq_col, place2)
        else:
            show_table(path, header, linker, seq_col, place1)


def mermaid(code: str) -> None:
    components.html(
        f"""
        
        <div style="height: 10vh; overflow: fit;">
            <pre class="mermaid">
                {code}
            </pre>
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """
    , height=500,scrolling=True)




def get_model_flow_chart(model_version: str) -> str:
    string = get_value_from_json(model_version, FLOW_CHART_PATH)
    mermaid(string)


def get_value_from_json(key: str, file_path: str = FLOW_CHART_PATH) -> str:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get(key, "Key not found in JSON")

