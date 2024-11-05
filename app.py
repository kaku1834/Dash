##-0 import-##
import streamlit as st
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.dates as mdates

st.set_page_config(
     page_title="Dashboard",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
)

##-1 Read csv and set cache-##
# Cache the data loading using @st.cache_data to improve performance
@st.cache_data
def load_data():
    # Load CSV files into Polars DataFrames
    raw = pl.read_csv('raw_add01.csv')
    stock = pl.read_csv('stock.csv')
    dateInfo = pl.read_csv('dateInfo.csv')
    
    # Convert the 'Date' column to datetime type in each DataFrame
    raw = raw.with_columns(pl.col('Date').str.strptime(pl.Datetime))
    stock = stock.with_columns(pl.col('Date').str.strptime(pl.Datetime))
    dateInfo = dateInfo.with_columns(pl.col('Date').str.strptime(pl.Datetime))
    
    return raw, stock, dateInfo

@st.cache_data  # 👈 Add the caching decorator
def get_sorted_unique_values(_df, column):
    return sorted(_df[column].unique())

# Load data
raw, stock, dateInfo = load_data()


##-2 Filter UI-##
sorted_Brand = get_sorted_unique_values(raw, 'Brand')
sorted_Region = get_sorted_unique_values(raw, 'Region')
sorted_Department = get_sorted_unique_values(raw, 'Department')
sorted_SubCategory = get_sorted_unique_values(raw, 'SubCategory')
sorted_Syuyaku = get_sorted_unique_values(raw, 'Syuyaku')
sorted_Tanpin = get_sorted_unique_values(raw, 'Tanpin')
sorted_Size = get_sorted_unique_values(raw, 'Size')
sorted_Color = get_sorted_unique_values(raw, 'Color')
sorted_SKU = get_sorted_unique_values(raw, 'SKU')

st.sidebar.header('Select department or category')
st.sidebar.caption('*There is an AND relation here')

selected_Brand = st.sidebar.multiselect('Select Brand', sorted_Brand, ['UQ', 'GU'])
selected_Region = st.sidebar.multiselect('Select Region', sorted_Region, ['JP', 'US'])
selected_Department = st.sidebar.multiselect('Select Department', sorted_Department, ['部門A', '部門B'])
selected_SubCategory = st.sidebar.multiselect('Select SubCategory', sorted_SubCategory, ['サブA', 'サブB'])
selected_Syuyaku = st.sidebar.multiselect('Select Syuyaku', sorted_Syuyaku, ['集約A', '集約B'])
selected_Tanpin = st.sidebar.multiselect('Select Tanpin', sorted_Tanpin, ['単品A', '単品B'])
selected_Size = st.sidebar.multiselect('Select Size', sorted_Size, ['S', 'M', 'L'])
selected_Color = st.sidebar.multiselect('Select Color', sorted_Color, ['Color1', 'Color2'])
selected_SKU = st.sidebar.multiselect('Select SKU', sorted_SKU, ['SColor1', 'SColor2', 'MColor1', 'MColor2', 'LColor1', 'LColor2'])

start_date = st.sidebar.date_input('Start date', raw.select(pl.col('Date')).min().to_series()[0])
end_date = st.sidebar.date_input('End date', raw.select(pl.col('Date')).max().to_series()[0])

@st.cache_data
def filter_data(_raw, selected_Brand, selected_Region, selected_Department, selected_SubCategory, selected_Syuyaku, selected_Tanpin, selected_Size, selected_Color, selected_SKU, start_date, end_date):
    conditions = []

    if len(selected_Brand) > 0:
        conditions.append(_raw['Brand'].is_in(selected_Brand))
    if len(selected_Region) > 0:
        conditions.append(_raw['Region'].is_in(selected_Region))
    if len(selected_Department) > 0:
        conditions.append(_raw['Department'].is_in(selected_Department))
    if len(selected_SubCategory) > 0:
        conditions.append(_raw['SubCategory'].is_in(selected_SubCategory))
    if len(selected_Syuyaku) > 0:
        conditions.append(_raw['Syuyaku'].is_in(selected_Syuyaku))
    if len(selected_Tanpin) > 0:
        conditions.append(_raw['Tanpin'].is_in(selected_Tanpin))
    if len(selected_Size) > 0:
        conditions.append(_raw['Size'].is_in(selected_Size))
    if len(selected_Color) > 0:
        conditions.append(_raw['Color'].is_in(selected_Color))
    if len(selected_SKU) > 0:
        conditions.append(_raw['SKU'].is_in(selected_SKU))
    if conditions:
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition &= condition
        df_01 = _raw.filter(combined_condition)
    else:
        df_01 = _raw.clone()
    if start_date or end_date:
        df_02 = df_01.filter((pl.col('Date') >= pl.lit(start_date)) & (pl.col('Date') <= pl.lit(end_date)))
    else:
        df_02 = df_01
    return df_02

raw_filtered = filter_data(raw, selected_Brand, selected_Region, selected_Department, selected_SubCategory, selected_Syuyaku, selected_Tanpin, selected_Size, selected_Color, selected_SKU, start_date, end_date)

##-3 Pre-process-##
# Processing raw sales daily data
raw_salesD = raw_filtered.group_by('Date').agg(pl.col('Sales').sum()).sort('Date')

# Merge raw sales data with dateInfo
raw_salesD_dateInfo = dateInfo.join(raw_salesD, on='Date', how='right')

# Calculate Weekly Data
raw_stockW = stock.group_by_dynamic('Date', every='1w').agg(pl.all().sum())

# Fixed Data
raw_salesD_dateInfo_1 = raw_salesD_dateInfo.select(['Date', 'Temperature', 'Event'])
# For calculate weekly data
raw_salesD_dateInfo_2 = raw_salesD_dateInfo.select(['Date', 'Sales', 'Customers'])

raw_salesD_dateInfo_2ed = raw_salesD_dateInfo_2.group_by_dynamic('Date', every='1w').agg(pl.all().sum())

# Combine data for final display
raw_salesW_dateInfo = raw_salesD_dateInfo_1.join(raw_salesD_dateInfo_2ed, on='Date', how='left').with_columns(
    (pl.col('Event').is_not_null()).cast(pl.Int8).alias('Holiday')
)

df_disp_pl = raw_salesW_dateInfo.join(raw_stockW, on='Date', how='left').with_columns([
    pl.col("Sales").fill_null(strategy="backward"),
    pl.col("Customers").fill_null(strategy="backward"),
    *[pl.col(f"{col}").fill_null(strategy="backward") for col in ["LColor1", "LColor2", "MColor1", "MColor2", "SColor1", "SColor2"]],
    pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date")
])

# Convert to pandas and show in Streamlit
df_disp = df_disp_pl.to_pandas()
df_disp['Date'] = pd.to_datetime(df_disp['Date'])
st.write("Final Processed Data:")


tab1, tab2 = st.tabs(["Dashborad", "Department"])

with tab1:
    st.title('Summary')
    st.subheader('KPI')
    col1, col2, col3 = st.columns(3)

    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")
    # @st.cache_data  # 👈 Add the caching decorator
    # def resample_weekly(df):
    #     return df.set_index('Date').resample('W').sum()

    # df_salesW = resample_weekly(df_salesD)
    st.subheader('Time Series Plot')

sku_cols = raw_filtered.select(pl.col("SKU")).unique().to_series().to_list()

color0 = '#B02A29'
color1 = '#00A1E4'
color2 = 'purple'
color3 = 'orange'
color4 = 'black'
color_box6 = ['#E4C087', '#F3F3E0', '#133E87', '#CBDCEB', '#BC7C7C', '#F6EFBD']

# Assuming df_disp and other data are already defined

fig, ax1 = plt.subplots(figsize=(16, 10), 
                            nrows=5, 
                            ncols=1, 
                            sharex=True, 
                            gridspec_kw={'hspace': 0},
                            height_ratios=(0.5, 1.5, 1.5, 1, 1)
                            )

# Plot Sales as a bar plot
df_disp_events = df_disp.dropna(subset=['Holiday']).query('Holiday == 1')
ax1[0].scatter(df_disp_events['Date'], df_disp_events['Holiday'], color=color0, label='Holiday')
ax1[0].set_ylabel('Holiday')
ax1[0].set_yticks([0, 1])
ax1[0].tick_params(axis='y')
ax1[0].legend(loc='upper right')
ax1[0].grid(True)

ax1[1].bar(df_disp['Date'], df_disp['Sales'], color=color1, label='Sales', alpha=0.5)
ax1[1].bar(df_disp.query('Event == Event')['Date'], df_disp.query('Event == Event')['Sales'], color=color0, label='Event Sales', alpha=0.8)
ax1[1].set_ylabel('Sales')
ax1[1].tick_params(axis='y')
ax1[1].legend(loc='upper right')

bar_width = 0.5
bottom_values = np.zeros(len(df_disp['Date']))

# Loop through all sku columns for the stacked bar plot
for i in range(len(sku_cols)):
    ax1[2].bar(df_disp['Date'], df_disp[sku_cols[i]], bottom=bottom_values,
            color=color_box6[i], edgecolor=color_box6[i], width=bar_width, label=sku_cols[i])
    bottom_values += df_disp[sku_cols[i]]

ax1[2].legend(loc='upper right')
ax1[2].set_ylabel('SKU')


# Create a third y-axis for Customers
ax1[3].plot(df_disp['Date'], df_disp['Customers'], color=color3, label='Customers', alpha=1, linewidth=2)
ax1[3].set_ylabel('Customers')
ax1[3].tick_params(axis='y')
ax1[3].legend(loc='upper right')

# Plot Temperature on the second subplot
ax1[4].plot(df_disp['Date'], df_disp['Temperature'], color=color4, label='Temperature', alpha=0.5)
ax1[4].set_ylabel('Temperature')
ax1[4].tick_params(axis='y')
ax1[4].legend(loc='upper right')

for ax in ax1:
    ax.label_outer()

# Create the second x-axis at the top
ax_top = ax1[0].twiny()  # Create a twin x-axis sharing the same x-axis as the top plot
ax_top.set_xlim(ax1[0].get_xlim())  # Match the x-limits

# Set major and minor ticks for the Date
ax_top.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks as months
ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format the date as 'Month Year'

# Optionally set label for the top axis
ax_top.set_xlabel('Date (Top Axis)')
ax_top.tick_params(axis='x', rotation=45)  # Rotate tick labels for readability

# Adjust the layout
fig.tight_layout()
st.pyplot(fig)


with tab2:
    st.title('Department')
    #  st.subheader('Sales')
    #  st.line_chart(df_disped, x = "Date", y = "Sales")
    #  st.subheader('Stocks')
    #  st.line_chart(df_disped, x = "Date", y = "Stocks")
    #  st.subheader('Customers')
    #  st.line_chart(df_disped, x = "Date", y = "Customers")
    #  st.subheader('Temperature')
    #  st.line_chart(df_disped, x = "Date", y = "Temperature")
##-0 import-##
import streamlit as st
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.dates as mdates

st.set_page_config(
     page_title="Dashboard",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
)

##-1 Read csv and set cache-##
# Cache the data loading using @st.cache_data to improve performance
@st.cache_data
def load_data():
    # Load CSV files into Polars DataFrames
    raw = pl.read_csv('raw_add01.csv')
    stock = pl.read_csv('stock.csv')
    dateInfo = pl.read_csv('dateInfo.csv')
    
    # Convert the 'Date' column to datetime type in each DataFrame
    raw = raw.with_columns(pl.col('Date').str.strptime(pl.Datetime))
    stock = stock.with_columns(pl.col('Date').str.strptime(pl.Datetime))
    dateInfo = dateInfo.with_columns(pl.col('Date').str.strptime(pl.Datetime))
    
    return raw, stock, dateInfo

@st.cache_data  # 👈 Add the caching decorator
def get_sorted_unique_values(_df, column):
    return sorted(_df[column].unique())

# Load data

# File uploader for CSV files
st.sidebar.header('Upload CSV files')
raw_file = st.sidebar.file_uploader("Upload raw_add01.csv", type=["csv"])
stock_file = st.sidebar.file_uploader("Upload stock.csv", type=["csv"])
date_info_file = st.sidebar.file_uploader("Upload dateInfo.csv", type=["csv"])

if raw_file and stock_file and date_info_file:
    
    raw, stock, dateInfo = load_data()


    ##-2 Filter UI-##
    sorted_Brand = get_sorted_unique_values(raw, 'Brand')
    sorted_Region = get_sorted_unique_values(raw, 'Region')
    sorted_Department = get_sorted_unique_values(raw, 'Department')
    sorted_SubCategory = get_sorted_unique_values(raw, 'SubCategory')
    sorted_Syuyaku = get_sorted_unique_values(raw, 'Syuyaku')
    sorted_Tanpin = get_sorted_unique_values(raw, 'Tanpin')
    sorted_Size = get_sorted_unique_values(raw, 'Size')
    sorted_Color = get_sorted_unique_values(raw, 'Color')
    sorted_SKU = get_sorted_unique_values(raw, 'SKU')

    st.sidebar.header('Select department or category')
    st.sidebar.caption('*There is an AND relation here')

    selected_Brand = st.sidebar.multiselect('Select Brand', sorted_Brand, ['UQ', 'GU'])
    selected_Region = st.sidebar.multiselect('Select Region', sorted_Region, ['JP', 'US'])
    selected_Department = st.sidebar.multiselect('Select Department', sorted_Department, ['部門A', '部門B'])
    selected_SubCategory = st.sidebar.multiselect('Select SubCategory', sorted_SubCategory, ['サブA', 'サブB'])
    selected_Syuyaku = st.sidebar.multiselect('Select Syuyaku', sorted_Syuyaku, ['集約A', '集約B'])
    selected_Tanpin = st.sidebar.multiselect('Select Tanpin', sorted_Tanpin, ['単品A', '単品B'])
    selected_Size = st.sidebar.multiselect('Select Size', sorted_Size, ['S', 'M', 'L'])
    selected_Color = st.sidebar.multiselect('Select Color', sorted_Color, ['Color1', 'Color2'])
    selected_SKU = st.sidebar.multiselect('Select SKU', sorted_SKU, ['SColor1', 'SColor2', 'MColor1', 'MColor2', 'LColor1', 'LColor2'])

    start_date = st.sidebar.date_input('Start date', raw.select(pl.col('Date')).min().to_series()[0])
    end_date = st.sidebar.date_input('End date', raw.select(pl.col('Date')).max().to_series()[0])

    @st.cache_data
    def filter_data(_raw, selected_Brand, selected_Region, selected_Department, selected_SubCategory, selected_Syuyaku, selected_Tanpin, selected_Size, selected_Color, selected_SKU, start_date, end_date):
        conditions = []

        if len(selected_Brand) > 0:
            conditions.append(_raw['Brand'].is_in(selected_Brand))
        if len(selected_Region) > 0:
            conditions.append(_raw['Region'].is_in(selected_Region))
        if len(selected_Department) > 0:
            conditions.append(_raw['Department'].is_in(selected_Department))
        if len(selected_SubCategory) > 0:
            conditions.append(_raw['SubCategory'].is_in(selected_SubCategory))
        if len(selected_Syuyaku) > 0:
            conditions.append(_raw['Syuyaku'].is_in(selected_Syuyaku))
        if len(selected_Tanpin) > 0:
            conditions.append(_raw['Tanpin'].is_in(selected_Tanpin))
        if len(selected_Size) > 0:
            conditions.append(_raw['Size'].is_in(selected_Size))
        if len(selected_Color) > 0:
            conditions.append(_raw['Color'].is_in(selected_Color))
        if len(selected_SKU) > 0:
            conditions.append(_raw['SKU'].is_in(selected_SKU))
        if conditions:
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition &= condition
            df_01 = _raw.filter(combined_condition)
        else:
            df_01 = _raw.clone()
        if start_date or end_date:
            df_02 = df_01.filter((pl.col('Date') >= pl.lit(start_date)) & (pl.col('Date') <= pl.lit(end_date)))
        else:
            df_02 = df_01
        return df_02

    raw_filtered = filter_data(raw, selected_Brand, selected_Region, selected_Department, selected_SubCategory, selected_Syuyaku, selected_Tanpin, selected_Size, selected_Color, selected_SKU, start_date, end_date)

    ##-3 Pre-process-##
    # Processing raw sales daily data
    raw_salesD = raw_filtered.group_by('Date').agg(pl.col('Sales').sum()).sort('Date')

    # Merge raw sales data with dateInfo
    raw_salesD_dateInfo = dateInfo.join(raw_salesD, on='Date', how='right')

    # Calculate Weekly Data
    raw_stockW = stock.group_by_dynamic('Date', every='1w').agg(pl.all().sum())

    # Fixed Data
    raw_salesD_dateInfo_1 = raw_salesD_dateInfo.select(['Date', 'Temperature', 'Event'])
    # For calculate weekly data
    raw_salesD_dateInfo_2 = raw_salesD_dateInfo.select(['Date', 'Sales', 'Customers'])

    raw_salesD_dateInfo_2ed = raw_salesD_dateInfo_2.group_by_dynamic('Date', every='1w').agg(pl.all().sum())

    # Combine data for final display
    raw_salesW_dateInfo = raw_salesD_dateInfo_1.join(raw_salesD_dateInfo_2ed, on='Date', how='left').with_columns(
        (pl.col('Event').is_not_null()).cast(pl.Int8).alias('Holiday')
    )

    df_disp_pl = raw_salesW_dateInfo.join(raw_stockW, on='Date', how='left').with_columns([
        pl.col("Sales").fill_null(strategy="backward"),
        pl.col("Customers").fill_null(strategy="backward"),
        *[pl.col(f"{col}").fill_null(strategy="backward") for col in ["LColor1", "LColor2", "MColor1", "MColor2", "SColor1", "SColor2"]],
        pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date")
    ])

    # Convert to pandas and show in Streamlit
    df_disp = df_disp_pl.to_pandas()
    df_disp['Date'] = pd.to_datetime(df_disp['Date'])
    st.write("Final Processed Data:")


    tab1, tab2 = st.tabs(["Dashborad", "Department"])

    with tab1:
        st.title('Summary')
        st.subheader('KPI')
        col1, col2, col3 = st.columns(3)

        col1.metric("Temperature", "70 °F", "1.2 °F")
        col2.metric("Wind", "9 mph", "-8%")
        col3.metric("Humidity", "86%", "4%")
        # @st.cache_data  # 👈 Add the caching decorator
        # def resample_weekly(df):
        #     return df.set_index('Date').resample('W').sum()

        # df_salesW = resample_weekly(df_salesD)
        st.subheader('Time Series Plot')

    sku_cols = raw_filtered.select(pl.col("SKU")).unique().to_series().to_list()

    color0 = '#B02A29'
    color1 = '#00A1E4'
    color2 = 'purple'
    color3 = 'orange'
    color4 = 'black'
    color_box6 = ['#E4C087', '#F3F3E0', '#133E87', '#CBDCEB', '#BC7C7C', '#F6EFBD']

    # Assuming df_disp and other data are already defined

    fig, ax1 = plt.subplots(figsize=(16, 10), 
                                nrows=5, 
                                ncols=1, 
                                sharex=True, 
                                gridspec_kw={'hspace': 0},
                                height_ratios=(0.5, 1.5, 1.5, 1, 1)
                                )

    # Plot Sales as a bar plot
    df_disp_events = df_disp.dropna(subset=['Holiday']).query('Holiday == 1')
    ax1[0].scatter(df_disp_events['Date'], df_disp_events['Holiday'], color=color0, label='Holiday')
    ax1[0].set_ylabel('Holiday')
    ax1[0].set_yticks([0, 1])
    ax1[0].tick_params(axis='y')
    ax1[0].legend(loc='upper right')
    ax1[0].grid(True)

    ax1[1].bar(df_disp['Date'], df_disp['Sales'], color=color1, label='Sales', alpha=0.5)
    ax1[1].bar(df_disp.query('Event == Event')['Date'], df_disp.query('Event == Event')['Sales'], color=color0, label='Event Sales', alpha=0.8)
    ax1[1].set_ylabel('Sales')
    ax1[1].tick_params(axis='y')
    ax1[1].legend(loc='upper right')

    bar_width = 0.5
    bottom_values = np.zeros(len(df_disp['Date']))

    # Loop through all sku columns for the stacked bar plot
    for i in range(len(sku_cols)):
        ax1[2].bar(df_disp['Date'], df_disp[sku_cols[i]], bottom=bottom_values,
                color=color_box6[i], edgecolor=color_box6[i], width=bar_width, label=sku_cols[i])
        bottom_values += df_disp[sku_cols[i]]

    ax1[2].legend(loc='upper right')
    ax1[2].set_ylabel('SKU')


    # Create a third y-axis for Customers
    ax1[3].plot(df_disp['Date'], df_disp['Customers'], color=color3, label='Customers', alpha=1, linewidth=2)
    ax1[3].set_ylabel('Customers')
    ax1[3].tick_params(axis='y')
    ax1[3].legend(loc='upper right')

    # Plot Temperature on the second subplot
    ax1[4].plot(df_disp['Date'], df_disp['Temperature'], color=color4, label='Temperature', alpha=0.5)
    ax1[4].set_ylabel('Temperature')
    ax1[4].tick_params(axis='y')
    ax1[4].legend(loc='upper right')

    for ax in ax1:
        ax.label_outer()

    # Create the second x-axis at the top
    ax_top = ax1[0].twiny()  # Create a twin x-axis sharing the same x-axis as the top plot
    ax_top.set_xlim(ax1[0].get_xlim())  # Match the x-limits

    # Set major and minor ticks for the Date
    ax_top.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks as months
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format the date as 'Month Year'

    # Optionally set label for the top axis
    ax_top.set_xlabel('Date (Top Axis)')
    ax_top.tick_params(axis='x', rotation=45)  # Rotate tick labels for readability

    # Adjust the layout
    fig.tight_layout()
    st.pyplot(fig)


    with tab2:
        st.title('Department')
        #  st.subheader('Sales')
        #  st.line_chart(df_disped, x = "Date", y = "Sales")
        #  st.subheader('Stocks')
        #  st.line_chart(df_disped, x = "Date", y = "Stocks")
        #  st.subheader('Customers')
        #  st.line_chart(df_disped, x = "Date", y = "Customers")
        #  st.subheader('Temperature')
        #  st.line_chart(df_disped, x = "Date", y = "Temperature")
