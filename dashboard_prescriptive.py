import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import base64

# === File Paths ===
REPORT_PATH = r"D:\projects\thermal_induction_motor\prescriptive maintenance\prescriptive_outputs\prescriptive_predictions.csv"
IMAGE_DIR = r"D:\projects\thermal_induction_motor\prescriptive maintenance\prescriptive_outputs"
HIGH_SEV_LOG_PATH = r"D:\projects\thermal_induction_motor\prescriptive maintenance\high_severity_log.csv"

# === Load CSV ===
def load_data():
    if os.path.exists(REPORT_PATH):
        df = pd.read_csv(REPORT_PATH)
        df.rename(columns={"Image": "Image_Name"}, inplace=True)  # Normalize column
        return df
    return pd.DataFrame()

# === Encode image to base64 ===
def encode_image(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    return None

# === Render card ===
def render_card(c):
    try:
        severity_val = float(c['Severity'])
    except:
        severity_val = 0

    alert = html.Div(
        "⚠️ High severity detected!",
        style={'color': 'red', 'fontWeight': 'bold', 'fontSize': '18px', 'marginBottom': '6px'}
    ) if severity_val >= 30 else None

    def line(text, size=15, margin=4):
        return html.Div(text, style={'fontSize': f'{size}px', 'marginBottom': f'{margin}px'})

    card_fields = [
        line(f"📸 Image: {c['Image_Name']}", 16),
        html.Img(src=c['Image_Encoded'], style={'width': '250px', 'margin': '10px'}),
        alert if alert else html.Div(),
        line(f"⚙️ Fault Type: {c['Fault_Type']}"),
        line(f"📊 Severity: {c['Severity']}%"),
        line(f"🔧 Action: {c['ActionTaken']}"),
        line(f"💰 Estimated Cost: ₹{round(c['Estimated_Cost'], 2)}"),
        line(f"⏱ Estimated Downtime: {round(c['Estimated_Downtime_Days'], 2)} days"),
    ]

    if severity_val >= 30:
        card_fields.extend([
            line(f"📌 Reason: {c.get('Reason', '')}"),
            line(f"🛠 Recommendation: {c.get('Recommendation', '')}"),
            line(f"🚨 Next Step: {c.get('Next_Step', '')}")
        ])

    return html.Div([
        html.Div(card_fields, style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'textAlign': 'center'
        }),
        html.Hr(style={'marginTop': '10px', 'marginBottom': '10px'})
    ], style={
        'border': '1px solid #ccc',
        'padding': '12px',
        'marginBottom': '15px',
        'borderRadius': '10px',
        'backgroundColor': '#fefefe',
        'boxShadow': '1px 1px 3px rgba(0,0,0,0.08)'
    })

# === Dash App ===
app = dash.Dash(__name__)
app.title = "Prescriptive Maintenance Dashboard"
server = app.server

# === Layout ===
app.layout = html.Div([
    html.H1("🛠 Real-Time Prescriptive Maintenance Log", style={'textAlign': 'center'}),
    html.Hr(),

    html.Div([
        html.Label("Filter:"),
        dcc.RadioItems(
            id='filter-option',
            options=[
                {'label': 'All', 'value': 'all'},
                {'label': 'High Severity Only', 'value': 'high'}
            ],
            value='all',
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '10px'}),

    dcc.Store(id='current-index', data=0),
    dcc.Store(id='shown-cards', data=[]),
    dcc.Interval(id='interval-update', interval=5000, n_intervals=0),
    html.Div(id='dashboard-content', style={'padding': '20px', 'maxHeight': '75vh', 'overflowY': 'scroll'}),
])

# === Callback ===
@app.callback(
    Output('dashboard-content', 'children'),
    Output('current-index', 'data'),
    Output('shown-cards', 'data'),
    Input('interval-update', 'n_intervals'),
    Input('filter-option', 'value'),
    State('current-index', 'data'),
    State('shown-cards', 'data')
)
def update_dashboard(n, filter_value, current_index, shown_cards_data):
    df = load_data()
    if df.empty:
        return [html.Div("⚠️ No data available.")], current_index, shown_cards_data

    if current_index >= len(df):
        visible_cards = [
            render_card(card) for card in shown_cards_data[::-1]
            if (filter_value == 'all' or float(card['Severity']) >= 30)
        ]
        done_message = html.Div("✅ All images have been shown.", style={
            'textAlign': 'center', 'color': 'green', 'fontSize': '20px', 'margin': '10px'
        })
        visible_cards.insert(0, done_message)
        return visible_cards, current_index, shown_cards_data

    row = df.iloc[current_index]
    img_src = encode_image(row['Image_Name'])

    try:
        severity = float(row['Severity'])
        if severity >= 30:
            if not os.path.exists(HIGH_SEV_LOG_PATH):
                row.to_frame().T.to_csv(HIGH_SEV_LOG_PATH, index=False)
            else:
                logged = pd.read_csv(HIGH_SEV_LOG_PATH)
                if row['Image_Name'] not in logged['Image_Name'].values:
                    row.to_frame().T.to_csv(HIGH_SEV_LOG_PATH, mode='a', header=False, index=False)
    except:
        pass

    new_card = {
        "Image_Name": row['Image_Name'],
        "Fault_Type": row['Fault_Type'],
        "Severity": row['Severity'],
        "ActionTaken": row['ActionTaken'],
        "Estimated_Cost": row['Estimated_Cost'],
        "Estimated_Downtime_Days": row['Estimated_Downtime_Days'],
        "Reason": row.get('Reason', ''),
        "Recommendation": row.get('Recommendation', ''),
        "Next_Step": row.get('Next_Step', ''),
        "Image_Encoded": img_src
    }
    shown_cards_data.append(new_card)

    filtered_cards = [
        render_card(c) for c in shown_cards_data[::-1]
        if (filter_value == 'all' or float(c['Severity']) >= 30)
    ]
    return filtered_cards, current_index + 1, shown_cards_data

# === Run ===
if __name__ == '__main__':
    app.run(debug=True)
