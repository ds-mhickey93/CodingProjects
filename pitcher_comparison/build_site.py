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

# ── Chart 0: Violin plots (SP vs RP distributions) — static matplotlib ───
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import base64, io

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Georgia', 'Cambria']

violin_df = df[['Pitcher_Type', 'IP', 'LI']].dropna().copy()
sp_data = violin_df[violin_df['Pitcher_Type'] == 'SP']
rp_data = violin_df[violin_df['Pitcher_Type'] == 'RP']

sp_color, rp_color_v = '#1B9E77', '#D95F02'
sp_pale, rp_pale = '#A3DFD0', '#F5C49A'

fig_v, (ax_ip, ax_li) = plt.subplots(1, 2, figsize=(14, 6))

def draw_violin_with_box(ax, datasets, positions, colors, pale_colors):
    parts = ax.violinplot(datasets, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(pale_colors[i])
        pc.set_edgecolor(colors[i])
        pc.set_alpha(0.45)
        pc.set_linewidth(0.8)
    box_width = 0.08
    for i, (data, pos) in enumerate(zip(datasets, positions)):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        median = np.median(data)
        ax.bar(pos, q3 - q1, bottom=q1, width=box_width,
               color='white', alpha=0.6, edgecolor='#555', linewidth=0.8, zorder=3)
        ax.hlines(median, pos - box_width / 2, pos + box_width / 2,
                  color='#333', linewidth=1.5, zorder=4)

# IP violin
draw_violin_with_box(ax_ip,
    [sp_data['IP'].values, rp_data['IP'].values],
    [1, 2], [sp_color, rp_color_v], [sp_pale, rp_pale])
ax_ip.set_xticks([1, 2])
ax_ip.set_xticklabels(['SP', 'RP'], fontsize=12)
ax_ip.set_ylabel('IP', fontsize=12, fontname='Georgia')
ax_ip.set_title('Innings Pitched (IP)', fontsize=14, fontname='Georgia', pad=8)
for spine in ax_ip.spines.values(): spine.set_visible(False)
ax_ip.tick_params(length=0)
ax_ip.yaxis.grid(True, color='#ddd', linewidth=0.5)
ax_ip.set_axisbelow(True)

# pLI violin
draw_violin_with_box(ax_li,
    [sp_data['LI'].values, rp_data['LI'].values],
    [1, 2], [sp_color, rp_color_v], [sp_pale, rp_pale])
ax_li.set_xticks([1, 2])
ax_li.set_xticklabels(['SP', 'RP'], fontsize=12)
ax_li.set_ylabel('pLI', fontsize=12, fontname='Georgia')
ax_li.set_title('Average Leverage Index (pLI)', fontsize=14, fontname='Georgia', pad=8)
for spine in ax_li.spines.values(): spine.set_visible(False)
ax_li.tick_params(length=0)
ax_li.yaxis.grid(True, color='#ddd', linewidth=0.5)
ax_li.set_axisbelow(True)

fig_v.suptitle(f'SP vs RP Distributions: Innings Pitched & pLI\nIndividual Pitcher-Seasons, {START_YEAR}\u2013{END_YEAR}',
               fontsize=18, fontweight='bold', fontname='Georgia', color='#333', y=1.02)
legend_elements = [Patch(facecolor=sp_pale, edgecolor=sp_color, alpha=0.45, label='SP'),
                   Patch(facecolor=rp_pale, edgecolor=rp_color_v, alpha=0.45, label='RP')]
fig_v.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=False)
plt.tight_layout()
plt.subplots_adjust(top=0.85)

buf = io.BytesIO()
fig_v.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_v)
buf.seek(0)
chart0_b64 = base64.b64encode(buf.read()).decode('utf-8')
chart0_html = f'<img src="data:image/png;base64,{chart0_b64}" alt="SP vs RP Violin Plots" style="width:100%;max-width:900px;display:block;margin:0 auto;">'

sp_median_ip = sp_data['IP'].median()
rp_median_ip = rp_data['IP'].median()
sp_median_li = sp_data['LI'].median()
rp_median_li = rp_data['LI'].median()

# ── Chart 1: Cluster scatter ─────────────────────────────────────────
rp_df['Cluster_Label'] = rp_df['Leverage_Cluster'].map(cluster_labels)
rp_df['ERA'] = rp_df['ERA'].round(2)
fig1 = px.scatter(
    rp_df, x='IP', y='LI',
    color='Cluster_Label',
    color_discrete_map={
        'Low Leverage': '#66A61E',
        'Medium Leverage': '#7570B3',
        'High Leverage': '#E7298A',
    },
    hover_data=['Player', 'ERA', 'WAR', 'LI', 'IP', 'Seasons'],
    labels={
        'IP': 'Innings Pitched (IP)',
        'LI': 'Average Leverage Index (pLI)',
        'Cluster_Label': 'Cluster',
    },
    title=f'K-Means Clustering of Relievers by pLI<br><sup>k = 3, {START_YEAR}\u2013{END_YEAR}</sup>',
)
fig1.update_traces(marker=dict(size=8, opacity=0.4))
fig1.update_layout(
    font_family='Georgia, Times New Roman, serif',
    height=550,
    legend_title_text='Leverage Tier',
    hoverlabel=dict(font_family='Georgia, Times New Roman, serif'),
    title_x=0.5,
    title_font_size=20,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#f9f9f9',
    margin=dict(l=60, r=30, t=80, b=60),
)
_hover_bg1 = {'Low Leverage': '#E8F5E0', 'Medium Leverage': '#ECEAF5', 'High Leverage': '#FCE4F0'}
fig1.for_each_trace(lambda t: t.update(
    hoverlabel=dict(bgcolor=_hover_bg1.get(t.name, '#fff'), font_color='#333'),
    hovertemplate=t.hovertemplate.replace('=', ' = ') if t.hovertemplate else t.hovertemplate
))
chart1_html = fig1.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

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
sp_color, rp_color = '#1B9E77', '#D95F02'

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
    height=700,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#f9f9f9',
    margin=dict(l=180, r=100, t=120, b=40),
)

# Style subplot titles
for ann in fig2.layout.annotations:
    ann.font = dict(size=15, family='Georgia', color='#444444')

fig2.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, zeroline=False, layer='above traces', row=1, col=1)
fig2.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1, zeroline=False, layer='above traces', row=2, col=1)
fig2.update_yaxes(showgrid=False, row=1, col=1)
fig2.update_yaxes(showgrid=False, row=2, col=1)
fig2.update_xaxes(range=[0, max(sp_total_wpa, rp_total_wpa) * 1.35], row=1, col=1)
fig2.update_xaxes(range=[0, sp_total_war * 1.45], row=2, col=1)

chart2_html = fig2.to_html(full_html=False, include_plotlyjs=False, config={'staticPlot': True, 'responsive': True})

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
        'LI': 'pLI',
    },
    title=f'WPA vs WAR: Starting Pitchers vs High-Leverage Relievers<br><sup>Individual Pitcher-Seasons, {START_YEAR}\u2013{END_YEAR}</sup>',
    color_discrete_map={'SP': '#1B9E77', 'RP (High Lev)': '#D95F02'},
)
fig3.update_traces(marker=dict(size=4, opacity=0.5))
fig3.update_layout(
    font_family='Georgia, Times New Roman, serif',
    height=650,
    hoverlabel=dict(font_family='Georgia, Times New Roman, serif'),
    title_x=0.5,
    title_font_size=20,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#f9f9f9',
    margin=dict(l=60, r=30, t=80, b=60),
)
_hover_bg3 = {'SP': '#D4F0E7', 'RP (High Lev)': '#FDEBD0'}
fig3.for_each_trace(lambda t: t.update(
    hoverlabel=dict(bgcolor=_hover_bg3.get(t.name, '#fff'), font_color='#333'),
    hovertemplate=t.hovertemplate.replace('=', ' = ') if t.hovertemplate else t.hovertemplate
))
chart3_html = fig3.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

# ── Cluster stats for the text ────────────────────────────────────────
cluster_stats = []
for cid in cluster_order:
    label = cluster_labels[cid]
    n = (rp_df['Leverage_Cluster'] == cid).sum()
    avg = rp_df[rp_df['Leverage_Cluster'] == cid]['LI'].mean()
    cluster_stats.append(f'{label}: {n} pitchers (avg pLI: {avg:.2f})')

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
<title>Reframing Bullpen Value: WAR, WPA, and the Leverage Gap</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  :root {{
    --bg: #fdfdfd;
    --text: #222;
    --muted: #555;
    --accent: #1B9E77;
    --accent2: #D95F02;
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
    max-width: 100%;
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
  .cluster-tag.high {{ background: #E7298A; }}
  .cluster-tag.med  {{ background: #7570B3; }}
  .cluster-tag.low  {{ background: #66A61E; }}

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
    <h1>Reframing Bullpen Value</h1>
    <p class="subtitle">
      WAR, WPA, and the Leverage Gap<br>
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
        <strong>pLI</strong> &mdash; Average Leverage Index
        <p>Quantifies the average importance of the game situations in which a pitcher is used. Leverage is determined by game flow and managerial deployment &mdash; not by the reliever himself.</p>
      </div>
    </div>

    <div class="callout">
      <p><strong>The Core Tension:</strong> WAR says high-leverage relievers are barely worth noticing. WPA says they're the most impactful pitchers in baseball. Neither is fully right.</p>
    </div>
  </div>
</section>

<!-- ── Data & Methodology ───────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>Data &amp; Methodology</h2>
    <p>
      All data is sourced from <strong>Fangraphs</strong> via the
      <a href="https://github.com/jldbc/pybaseball" style="color: var(--accent);">pybaseball</a>
      Python library, covering MLB seasons from <strong>{START_YEAR}&ndash;{END_YEAR}</strong>.
      A minimum threshold of <strong>{QUAL} innings pitched</strong> per season is applied to filter out
      position-player pitching appearances and other negligible outings.
    </p>

    <h3>WAR Variant: fWAR</h3>
    <p>
      The WAR values used throughout this analysis are <strong>fWAR</strong> (Fangraphs Wins Above Replacement).
      Unlike Baseball-Reference&rsquo;s bWAR, which is built on RA9 (runs allowed per 9 innings),
      fWAR uses <strong>FIP</strong> (Fielding Independent Pitching) as its core performance component.
      FIP isolates strikeouts, walks, hit-by-pitches, and home runs &mdash; outcomes the pitcher directly controls &mdash;
      stripping out batted-ball luck and defensive quality. This makes fWAR a better measure of repeatable
      pitcher skill, but it also means that any value derived from inducing weak contact or suppressing
      BABIP is not captured.
    </p>

    <h3>Pitcher Classification</h3>
    <div class="metrics">
      <div class="metric-card">
        <strong>Starting Pitcher (SP)</strong>
        <p>Games Started &ge; 5 <em>and</em> Innings Pitched &ge; 20 in a season. This captures pitchers with a meaningful starting workload while excluding openers and spot starters with minimal usage.</p>
      </div>
      <div class="metric-card">
        <strong>Relief Pitcher (RP)</strong>
        <p>Games &ge; 5 <em>and</em> Games Started &lt; 3 in a season. The low GS ceiling excludes swingmen and spot starters, isolating pitchers used primarily in relief. Pitchers who don&rsquo;t meet either definition are excluded.</p>
      </div>
    </div>
  </div>
</section>

<!-- ── SP/RP Divide ──────────────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>The SP/RP Divide: Innings and pLI</h2>
    <p>
      Before identifying high-leverage relievers, it&rsquo;s worth seeing just how differently
      starters and relievers are deployed. The violin plots below show the distributions of
      <strong>Innings Pitched</strong> and <strong>pLI</strong> across all individual
      pitcher-seasons &mdash; not aggregated careers, but each season as its own data point.
    </p>

    <div class="stat-row">
      <div class="stat">
        <span class="number">{sp_median_ip:.0f}</span>
        <span class="label">SP Median IP</span>
      </div>
      <div class="stat">
        <span class="number">{rp_median_ip:.0f}</span>
        <span class="label">RP Median IP</span>
      </div>
      <div class="stat">
        <span class="number">{sp_median_li:.2f}</span>
        <span class="label">SP Median pLI</span>
      </div>
      <div class="stat">
        <span class="number">{rp_median_li:.2f}</span>
        <span class="label">RP Median pLI</span>
      </div>
    </div>

    <div class="chart-wrap">{chart0_html}</div>

    <h3>Volume vs. Leverage</h3>
    <p>
      Starting pitchers dominate the workload side of the equation &mdash; their IP distribution
      stretches far above anything relievers produce, reflecting the fundamental design of the role.
      A typical SP season covers three to four times the innings of a typical RP season, which is
      precisely why WAR favors them so heavily.
    </p>

    <h3>Leverage Concentration</h3>
    <p>
      While starting pitchers cluster tightly around a league-average pLI (~1.0), relievers show
      enormous variance. A substantial group of relievers consistently operate well above that
      baseline &mdash; these are the high-leverage arms deployed in the highest-stakes situations.
    </p>

    <div class="callout">
      <p>Consider the asymmetry: a starter throwing six strong innings in a comfortable lead
      may shift win probability by only a few percentage points. A reliever striking out two
      batters with the tying run on third in the eighth inning can swing win probability by
      30&ndash;40% in two minutes. Both performances require real skill &mdash; but WPA rewards
      the reliever&rsquo;s two minutes far more than the starter&rsquo;s six innings, not because
      the reliever is better, but because the <em>situation</em> carried more weight.</p>
    </div>
  </div>
</section>

<!-- ── Clustering ────────────────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>Identifying High-Leverage Relievers</h2>
    <p>
      We aggregate each pitcher's career totals across all seasons, then use k-means clustering (k&nbsp;=&nbsp;3)
      on average pLI to separate relievers into three tiers.
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
        <span class="label">Avg pLI (High-Lev)</span>
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
