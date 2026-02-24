"""
Build a self-contained HTML website from the pitcher value analysis.
Reads the cached CSV, runs the analysis, and writes index.html.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

# ── Settings ──────────────────────────────────────────────────────────
START_YEAR = 2016
END_YEAR = 2025
QUAL = 5
CSV_PATH = f'fangraphs_pitchers_{START_YEAR}_{END_YEAR}_IP{QUAL}.csv'
OUT_PATH = 'index.html'

# ── Load & classify ──────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

rename_map = {
    'Name': 'Player', 'K%': 'Strikeout_Rate', 'BB%': 'Walk_Rate',
    'pLI': 'LI', 'WPA/LI': 'WPA_LI', 'SO': 'Strikeouts', 'BB': 'Walks',
    'G': 'Games', 'GS': 'Games_Started', 'FIP': 'FIP', 'ERA': 'ERA',
    'WAR': 'WAR', 'WPA': 'WPA', 'WHIP': 'WHIP'
}
df = df.rename(columns=rename_map)


def classify_pitcher(row):
    if row.get('Games_Started', 0) >= 5 and row.get('IP', 0) >= 20:
        return 'SP'
    elif row.get('Games', 0) >= 5 and row.get('Games_Started', 0) < 3:
        return 'RP'
    return 'Other'


df['Pitcher_Type'] = df.apply(classify_pitcher, axis=1)
df = df[df['Pitcher_Type'].isin(['SP', 'RP'])].reset_index(drop=True)

for col in ['IP', 'LI', 'FIP', 'ERA', 'WAR', 'WPA', 'WHIP', 'Games', 'Games_Started']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Aggregate & cluster ──────────────────────────────────────────────
agg_dict = {
    'IP': 'sum', 'WPA': 'sum', 'WAR': 'sum', 'Games': 'sum', 'Games_Started': 'sum',
    'ERA': 'mean', 'FIP': 'mean', 'WHIP': 'mean', 'LI': 'mean',
    'Pitcher_Type': 'first', 'Season': 'count'
}
df_agg = df.groupby('Player').agg(agg_dict).reset_index()
df_agg.rename(columns={'Season': 'Seasons'}, inplace=True)

rp_df = df_agg[df_agg['Pitcher_Type'] == 'RP'].copy().dropna(subset=['IP', 'LI'])
kmeans = KMeans(n_clusters=3, random_state=42)
rp_df['Leverage_Cluster'] = kmeans.fit_predict(rp_df[['LI']].values)

cluster_means = rp_df.groupby('Leverage_Cluster')['LI'].mean()
high_leverage_cluster = cluster_means.idxmax()
rp_df['High_Leverage'] = rp_df['Leverage_Cluster'] == high_leverage_cluster

sp_df = df_agg[df_agg['Pitcher_Type'] == 'SP'].copy()
high_lev_rp_df = rp_df[rp_df['High_Leverage']].copy()
high_lev_rp_df['Pitcher_Type'] = 'RP (High Lev)'

comparison_df = pd.concat([sp_df, high_lev_rp_df], ignore_index=True)
comparison_df = comparison_df.dropna(subset=['ERA', 'IP', 'LI'])

# ── Cluster labels ────────────────────────────────────────────────────
cluster_li = rp_df.groupby('Leverage_Cluster')['LI'].mean().sort_values()
cluster_order = list(cluster_li.index)
cluster_labels = {
    cluster_order[0]: 'Low Leverage',
    cluster_order[1]: 'Medium Leverage',
    cluster_order[2]: 'High Leverage',
}

# ── Chart 1: Cluster scatter ─────────────────────────────────────────
rp_df['Cluster_Label'] = rp_df['Leverage_Cluster'].map(cluster_labels)
fig1 = px.scatter(
    rp_df, x='IP', y='LI',
    color='Cluster_Label',
    color_discrete_map={
        'Low Leverage': '#BCBD22',
        'Medium Leverage': '#FF7F0E',
        'High Leverage': '#D62728',
    },
    hover_data=['Player', 'ERA', 'WAR', 'LI', 'IP', 'Seasons'],
    labels={
        'IP': 'Innings Pitched (IP)',
        'LI': 'Average Leverage Index (LI)',
        'Cluster_Label': 'Cluster',
    },
    title=f'K-Means Clustering of Relievers by Leverage Index<br><sup>k = 3, {START_YEAR}\u2013{END_YEAR}</sup>',
)
fig1.update_traces(marker=dict(size=8, opacity=0.4))
fig1.update_layout(
    font_family='Georgia, Times New Roman, serif',
    height=550, width=900,
    legend_title_text='Leverage Tier',
    hoverlabel=dict(font_family='Georgia, Times New Roman, serif'),
    title_x=0.5,
    title_font_size=20,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#f9f9f9',
    margin=dict(l=60, r=30, t=80, b=60),
)
fig1.for_each_trace(lambda t: t.update(
    hovertemplate=t.hovertemplate.replace('=', ' = ') if t.hovertemplate else t.hovertemplate
))
chart1_html = fig1.to_html(full_html=False, include_plotlyjs=False)

# ── Chart 2: Bar chart (recreated in Plotly) ──────────────────────────
comparison_wpa = comparison_df.dropna(subset=['WPA'])
comparison_war = comparison_df.dropna(subset=['WAR'])

sp_total_wpa = comparison_wpa[comparison_wpa['Pitcher_Type'] == 'SP']['WPA'].sum()
rp_total_wpa = comparison_wpa[comparison_wpa['Pitcher_Type'] == 'RP (High Lev)']['WPA'].sum()
sp_total_war = comparison_war[comparison_war['Pitcher_Type'] == 'SP']['WAR'].sum()
rp_total_war = comparison_war[comparison_war['Pitcher_Type'] == 'RP (High Lev)']['WAR'].sum()

from plotly.subplots import make_subplots

fig2 = make_subplots(
    rows=2, cols=1,
    subplot_titles=['Win Probability Added (WPA)', 'Wins Above Replacement (WAR)'],
    vertical_spacing=0.22,
)

categories = ['Starting Pitchers', 'High-Leverage Relievers']
sp_color, rp_color = '#0072B2', '#E69F00'

# WPA bars
fig2.add_trace(go.Bar(
    y=categories, x=[sp_total_wpa, rp_total_wpa],
    orientation='h',
    marker_color=[sp_color, rp_color],
    opacity=0.85,
    text=[f'{sp_total_wpa:.1f}', f'{rp_total_wpa:.1f}'],
    textposition='outside',
    textfont=dict(size=13, family='Georgia'),
    hovertemplate='%{y}: %{x:.1f} Wins Added<extra></extra>',
    showlegend=False,
    width=0.45,
), row=1, col=1)

# WAR bars
fig2.add_trace(go.Bar(
    y=categories, x=[sp_total_war, rp_total_war],
    orientation='h',
    marker_color=[sp_color, rp_color],
    opacity=0.85,
    text=[f'{sp_total_war:.1f}', f'{rp_total_war:.1f}'],
    textposition='outside',
    textfont=dict(size=13, family='Georgia'),
    hovertemplate='%{y}: %{x:.1f} WAR<extra></extra>',
    showlegend=False,
    width=0.45,
), row=2, col=1)

fig2.update_layout(
    title=dict(
        text=f'Two Metrics, Two Stories<br><sup>How WAR and WPA See High-Leverage Relievers ({START_YEAR}\u2013{END_YEAR})</sup>',
        font=dict(size=22, family='Georgia', color='#333333'),
        x=0.5,
    ),
    font_family='Georgia, Times New Roman, serif',
    height=700, width=900,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#f9f9f9',
    margin=dict(l=180, r=100, t=120, b=40),
)

# Style subplot titles
for ann in fig2.layout.annotations:
    ann.font = dict(size=15, family='Georgia', color='#444444')

fig2.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, zeroline=False, row=1, col=1)
fig2.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, zeroline=False, row=2, col=1)
fig2.update_yaxes(showgrid=False, row=1, col=1)
fig2.update_yaxes(showgrid=False, row=2, col=1)
fig2.update_xaxes(range=[0, max(sp_total_wpa, rp_total_wpa) * 1.35], row=1, col=1)
fig2.update_xaxes(range=[0, sp_total_war * 1.45], row=2, col=1)

chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

# ── Chart 3: WPA vs WAR scatter ───────────────────────────────────────
hl_rp_names = high_lev_rp_df['Player'].unique()

scatter_df = df[['Player', 'Pitcher_Type', 'Season', 'WAR', 'WPA', 'ERA', 'IP', 'LI']].dropna(subset=['WAR', 'WPA']).copy()
scatter_df = scatter_df[
    (scatter_df['Pitcher_Type'] == 'SP') |
    ((scatter_df['Pitcher_Type'] == 'RP') & (scatter_df['Player'].isin(hl_rp_names)))
].copy()
scatter_df['Pitcher_Type'] = scatter_df['Pitcher_Type'].replace({'RP': 'RP (High Lev)'})

for col in ['WAR', 'WPA', 'ERA', 'IP', 'LI']:
    scatter_df[col] = scatter_df[col].round(2)

fig3 = px.scatter(
    scatter_df, x='WAR', y='WPA', color='Pitcher_Type',
    hover_data=['Player', 'Season', 'ERA', 'IP', 'LI', 'WAR', 'WPA'],
    labels={
        'WAR': 'Wins Above Replacement (WAR)',
        'WPA': 'Win Probability Added (WPA)',
    },
    title=f'WPA vs WAR: Starting Pitchers vs High-Leverage Relievers<br><sup>Individual Pitcher-Seasons, {START_YEAR}\u2013{END_YEAR}</sup>',
    color_discrete_map={'SP': '#0072B2', 'RP (High Lev)': '#E69F00'},
)
fig3.update_traces(marker=dict(size=4, opacity=0.5))
fig3.update_layout(
    font_family='Georgia, Times New Roman, serif',
    height=650, width=950,
    hoverlabel=dict(font_family='Georgia, Times New Roman, serif'),
    title_x=0.5,
    title_font_size=20,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#f9f9f9',
    margin=dict(l=60, r=30, t=80, b=60),
)
fig3.for_each_trace(lambda t: t.update(
    hovertemplate=t.hovertemplate.replace('=', ' = ') if t.hovertemplate else t.hovertemplate
))
chart3_html = fig3.to_html(full_html=False, include_plotlyjs=False)

# ── Cluster stats for the text ────────────────────────────────────────
cluster_stats = []
for cid in cluster_order:
    label = cluster_labels[cid]
    n = (rp_df['Leverage_Cluster'] == cid).sum()
    avg = rp_df[rp_df['Leverage_Cluster'] == cid]['LI'].mean()
    cluster_stats.append(f'{label}: {n} pitchers (avg LI: {avg:.2f})')

wpa_ratio = (rp_total_wpa / sp_total_wpa) * 100
war_ratio = (rp_total_war / sp_total_war) * 100

n_sp = len(sp_df)
n_rp = len(rp_df)
n_hl = len(high_lev_rp_df)
avg_hl_li = high_lev_rp_df['LI'].mean()
n_total = len(df)

# ── Build HTML ────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Valuation Gap: High-Leverage Relievers</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  :root {{
    --bg: #fdfdfd;
    --text: #222;
    --muted: #555;
    --accent: #0072B2;
    --accent2: #E69F00;
    --border: #e0e0e0;
    --card-bg: #ffffff;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: Georgia, 'Times New Roman', serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
  }}

  /* ── Hero ─────────────────────────────────────── */
  .hero {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #f0f0f0;
    padding: 5rem 2rem 4rem;
    text-align: center;
  }}
  .hero h1 {{
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
    line-height: 1.2;
  }}
  .hero .subtitle {{
    font-size: 1.15rem;
    color: #bbb;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.6;
  }}

  /* ── Container ────────────────────────────────── */
  .container {{
    max-width: 960px;
    margin: 0 auto;
    padding: 0 2rem;
  }}

  /* ── Sections ─────────────────────────────────── */
  section {{
    padding: 3.5rem 0;
  }}
  section + section {{
    border-top: 1px solid var(--border);
  }}

  h2 {{
    font-size: 1.8rem;
    color: #333;
    margin-bottom: 1rem;
    font-weight: 700;
    letter-spacing: -0.3px;
  }}
  h3 {{
    font-size: 1.25rem;
    color: #444;
    margin-bottom: 0.6rem;
    font-weight: 600;
  }}

  p {{
    margin-bottom: 1rem;
    color: var(--muted);
    font-size: 1.05rem;
  }}

  /* ── Metrics grid ─────────────────────────────── */
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.2rem;
    margin: 1.5rem 0;
  }}
  .metric-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
  }}
  .metric-card strong {{
    color: var(--text);
    font-size: 1rem;
  }}
  .metric-card p {{
    margin: 0.3rem 0 0;
    font-size: 0.95rem;
    color: #666;
  }}

  /* ── Key stat callouts ────────────────────────── */
  .callout {{
    background: #f4f7fa;
    border-left: 4px solid var(--accent);
    padding: 1.2rem 1.5rem;
    margin: 1.5rem 0;
    border-radius: 0 6px 6px 0;
  }}
  .callout p {{
    margin: 0;
    color: #333;
    font-size: 1rem;
  }}

  /* ── Stat badges ──────────────────────────────── */
  .stat-row {{
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin: 1rem 0;
  }}
  .stat {{
    text-align: center;
    flex: 1;
    min-width: 140px;
  }}
  .stat .number {{
    display: block;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
  }}
  .stat .label {{
    font-size: 0.85rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.3rem;
    display: block;
  }}

  /* ── Chart wrapper ────────────────────────────── */
  .chart-wrap {{
    margin: 2rem auto;
    max-width: 960px;
    overflow-x: auto;
  }}
  .chart-wrap > div {{
    margin: 0 auto;
  }}

  /* ── Cluster stats ────────────────────────────── */
  .cluster-stats {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
  }}
  .cluster-tag {{
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #fff;
  }}
  .cluster-tag.high {{ background: #D62728; }}
  .cluster-tag.med  {{ background: #FF7F0E; }}
  .cluster-tag.low  {{ background: #BCBD22; color: #333; }}

  /* ── Conclusion ───────────────────────────────── */
  .conclusion {{
    background: #fafafa;
    padding: 3rem 0;
  }}
  .conclusion ul {{
    list-style: none;
    padding: 0;
  }}
  .conclusion li {{
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 1.05rem;
    color: var(--muted);
  }}
  .conclusion li:last-child {{
    border-bottom: none;
  }}
  .conclusion li strong {{
    color: var(--text);
  }}

  .takeaway {{
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: #e8e8e8;
    padding: 2rem;
    border-radius: 10px;
    margin-top: 1.5rem;
    text-align: center;
    font-size: 1.1rem;
    line-height: 1.6;
  }}
  .takeaway strong {{
    color: #fff;
  }}

  /* ── Footer ───────────────────────────────────── */
  footer {{
    text-align: center;
    padding: 2rem;
    color: #999;
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
  }}

  /* ── Responsive ───────────────────────────────── */
  @media (max-width: 640px) {{
    .hero h1 {{ font-size: 2rem; }}
    .hero {{ padding: 3rem 1.5rem 2.5rem; }}
    .container {{ padding: 0 1.2rem; }}
    .stat .number {{ font-size: 1.7rem; }}
    h2 {{ font-size: 1.5rem; }}
  }}
</style>
</head>
<body>

<!-- ── Hero ──────────────────────────────────────────────────────── -->
<header class="hero">
  <div class="container">
    <h1>The Valuation Gap</h1>
    <p class="subtitle">
      Where Do High-Leverage Relievers Really Stand?<br>
      An analysis of {n_total:,} pitcher-seasons from {START_YEAR}&ndash;{END_YEAR}
    </p>
  </div>
</header>

<!-- ── Intro ─────────────────────────────────────────────────────── -->
<section>
  <div class="container">
    <p>
      High-leverage relievers sit in a valuation gap between baseball's two primary value metrics.
      <strong>WAR systematically undervalues</strong> elite relievers while
      <strong>WPA systematically overvalues</strong> them &mdash; and the truth lies somewhere in between.
    </p>

    <div class="metrics">
      <div class="metric-card">
        <strong>WAR</strong> &mdash; Wins Above Replacement
        <p>Measures pitcher value via performance (ERA/FIP) &times; volume (IP). Isolates underlying skill but <em>ignores leverage</em>, treating all innings as equally important.</p>
      </div>
      <div class="metric-card">
        <strong>WPA</strong> &mdash; Win Probability Added
        <p>Measures actual impact on win probability, fully incorporating timing and context. But it absorbs <em>managerial usage, sequencing luck, and situational noise</em> the pitcher doesn't control.</p>
      </div>
      <div class="metric-card">
        <strong>LI</strong> &mdash; Leverage Index
        <p>Quantifies the importance of the game situation when a pitcher enters. Leverage is determined by game flow and managerial deployment &mdash; not by the reliever himself.</p>
      </div>
    </div>

    <div class="callout">
      <p><strong>The Core Tension:</strong> WAR says high-leverage relievers are barely worth noticing. WPA says they're the most impactful pitchers in baseball. Neither is fully right.</p>
    </div>
  </div>
</section>

<!-- ── Clustering ────────────────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>Identifying High-Leverage Relievers</h2>
    <p>
      We aggregate each pitcher's career totals across all seasons, then use k-means clustering (k&nbsp;=&nbsp;3)
      on average Leverage Index to separate relievers into three tiers.
    </p>

    <div class="stat-row">
      <div class="stat">
        <span class="number">{n_sp}</span>
        <span class="label">Starting Pitchers</span>
      </div>
      <div class="stat">
        <span class="number">{n_rp}</span>
        <span class="label">All Relievers</span>
      </div>
      <div class="stat">
        <span class="number">{n_hl}</span>
        <span class="label">High-Leverage RPs</span>
      </div>
      <div class="stat">
        <span class="number">{avg_hl_li:.2f}</span>
        <span class="label">Avg LI (High-Lev)</span>
      </div>
    </div>

    <div class="cluster-stats">
      <span class="cluster-tag low">{cluster_stats[0]}</span>
      <span class="cluster-tag med">{cluster_stats[1]}</span>
      <span class="cluster-tag high">{cluster_stats[2]}</span>
    </div>

    <div class="chart-wrap">{chart1_html}</div>
  </div>
</section>

<!-- ── The Divergence ────────────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>Two Metrics, Two Stories</h2>
    <p>
      By WPA, high-leverage relievers outproduce all starting pitchers combined.
      By WAR, they're worth a fraction. Neither extreme reflects reality &mdash;
      WAR ignores that these innings matter more, while WPA inflates them with
      sequencing and managerial context the pitcher doesn't control.
    </p>

    <div class="stat-row">
      <div class="stat">
        <span class="number">{wpa_ratio:.0f}%</span>
        <span class="label">HL RPs as % of SP WPA</span>
      </div>
      <div class="stat">
        <span class="number">{war_ratio:.0f}%</span>
        <span class="label">HL RPs as % of SP WAR</span>
      </div>
    </div>

    <div class="chart-wrap">{chart2_html}</div>
  </div>
</section>

<!-- ── Scatter ───────────────────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>Individual Seasons: WPA vs WAR</h2>
    <p>
      Each dot is a single pitcher-season. SPs spread horizontally (high WAR, moderate WPA) &mdash;
      their volume drives skill-based value but dilutes game-context impact.
      High-leverage RPs spread vertically (low WAR, wide WPA range) &mdash;
      their value is concentrated in high-stakes moments but inflated by
      sequencing and managerial deployment.
    </p>
    <p>
      The gap between the two clusters is the valuation no-man's-land where
      neither metric tells the full story.
    </p>

    <div class="chart-wrap">{chart3_html}</div>
  </div>
</section>

<!-- ── Conclusion ────────────────────────────────────────────────── -->
<section class="conclusion">
  <div class="container">
    <h2>Conclusion</h2>
    <ul>
      <li>
        <strong>WAR</strong> is designed to credit total volume of performance and isolate underlying skill.
        This leads it to systematically <strong>undervalue</strong> relievers by treating all innings as equally
        important and ignoring leverage.
      </li>
      <li>
        <strong>WPA</strong> measures who actually swings game outcomes, fully incorporating timing and context.
        This causes it to systematically <strong>overvalue</strong> relievers by absorbing managerial usage,
        sequencing, and luck.
      </li>
      <li>
        <strong>Leverage</strong> itself is not controlled by relievers &mdash; it is determined by game flow and
        managerial deployment. Yet every team generates a large number of high-leverage moments over a season,
        meaning elite relief skill reliably translates into real wins.
      </li>
    </ul>

    <div class="takeaway">
      The true value of high-leverage relievers lies <strong>between the implications of WAR and WPA</strong>:
      higher than WAR suggests, but substantially lower than raw WPA implies.
      They play an outsized role in converting win probability into actual victories &mdash;
      without overstating their underlying contribution to total run prevention.
    </div>
  </div>
</section>

<footer>
  <div class="container">
    Data: Fangraphs via pybaseball &middot; {START_YEAR}&ndash;{END_YEAR} &middot; Min {QUAL} IP
  </div>
</footer>

</body>
</html>
"""

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Built {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)')
