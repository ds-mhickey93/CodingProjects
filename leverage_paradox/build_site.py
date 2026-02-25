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
    cluster_order[1]: 'Mid-Leverage',
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

def draw_violin_with_strip(ax, datasets, positions, colors, pale_colors, jitter_width=0.25):
    rng = np.random.default_rng(42)
    parts = ax.violinplot(datasets, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(pale_colors[i])
        pc.set_edgecolor(colors[i])
        pc.set_alpha(0.55)
        pc.set_linewidth(0.8)
        pc.set_zorder(3)
    for i, (data, pos) in enumerate(zip(datasets, positions)):
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(data))
        ax.scatter(pos + jitter, data, s=4, alpha=0.25, color=colors[i],
                   edgecolors='none', zorder=1, rasterized=True)
    box_width = 0.08
    for i, (data, pos) in enumerate(zip(datasets, positions)):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        median = np.median(data)
        ax.bar(pos, q3 - q1, bottom=q1, width=box_width,
               color='white', alpha=0.7, edgecolor='#555', linewidth=0.8, zorder=5)
        ax.hlines(median, pos - box_width / 2, pos + box_width / 2,
                  color='#333', linewidth=1.5, zorder=6)

# IP violin
draw_violin_with_strip(ax_ip,
    [sp_data['IP'].values, rp_data['IP'].values],
    [1, 2], [sp_color, rp_color_v], [sp_pale, rp_pale])
ax_ip.set_xticks([1, 2])
ax_ip.set_xticklabels(['SP', 'RP'], fontsize=13)
ax_ip.set_ylabel('IP', fontsize=13, fontname='Georgia')
ax_ip.set_title('Innings Pitched (IP)', fontsize=16, fontname='Georgia', pad=8)
for spine in ax_ip.spines.values(): spine.set_visible(False)
ax_ip.tick_params(length=0)
ax_ip.yaxis.grid(True, color='#ddd', linewidth=0.5)
ax_ip.set_axisbelow(True)

# pLI violin
draw_violin_with_strip(ax_li,
    [sp_data['LI'].values, rp_data['LI'].values],
    [1, 2], [sp_color, rp_color_v], [sp_pale, rp_pale])
ax_li.set_xticks([1, 2])
ax_li.set_xticklabels(['SP', 'RP'], fontsize=13)
ax_li.set_ylabel('pLI', fontsize=13, fontname='Georgia')
ax_li.set_title('Average Leverage Index (pLI)', fontsize=16, fontname='Georgia', pad=8)
for spine in ax_li.spines.values(): spine.set_visible(False)
ax_li.tick_params(length=0)
ax_li.yaxis.grid(True, color='#ddd', linewidth=0.5)
ax_li.set_axisbelow(True)

fig_v.suptitle(f'SP vs RP Distributions: Innings Pitched & pLI\n{START_YEAR}\u2013{END_YEAR} Individual Pitcher-Seasons',
               fontsize=17, fontweight='normal', fontname='Georgia', color='#444', y=1.02)
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
rp_df['WAR'] = rp_df['WAR'].round(2)
rp_df['LI'] = rp_df['LI'].round(2)
rp_df['IP'] = rp_df['IP'].round(1)
fig1 = px.scatter(
    rp_df, x='IP', y='LI',
    color='Cluster_Label',
    color_discrete_map={
        'Low Leverage': '#66A61E',
        'Mid-Leverage': '#7570B3',
        'High Leverage': '#D64541',
    },
    category_orders={'Cluster_Label': ['High Leverage', 'Mid-Leverage', 'Low Leverage']},
    hover_data=['Player', 'ERA', 'WAR', 'LI', 'IP', 'Seasons'],
    labels={
        'IP': 'Innings Pitched (IP)',
        'LI': 'Average Leverage Index (pLI)',
        'Cluster_Label': 'Cluster',
    },
    title=f'Relief Pitcher Leverage Tiers<br><sup>K-Means Clustering by pLI (k = 3), {START_YEAR}\u2013{END_YEAR}</sup>',
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
_hover_bg1 = {'Low Leverage': '#E8F5E0', 'Mid-Leverage': '#ECEAF5', 'High Leverage': '#FDEAE9'}
fig1.for_each_trace(lambda t: t.update(
    hoverlabel=dict(bgcolor=_hover_bg1.get(t.name, '#fff'), font_color='#333'),
    hovertemplate=t.hovertemplate.replace('=', ' = ') if t.hovertemplate else t.hovertemplate
))
chart1_html = fig1.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

# ── Chart 2: Bar chart — static matplotlib ────────────────────────────
comparison_wpa = comparison_df.dropna(subset=['WPA'])
comparison_war = comparison_df.dropna(subset=['WAR'])

sp_total_wpa = comparison_wpa[comparison_wpa['Pitcher_Type'] == 'SP']['WPA'].sum()
rp_total_wpa = comparison_wpa[comparison_wpa['Pitcher_Type'] == 'RP (High Lev)']['WPA'].sum()
sp_total_war = comparison_war[comparison_war['Pitcher_Type'] == 'SP']['WAR'].sum()
rp_total_war = comparison_war[comparison_war['Pitcher_Type'] == 'RP (High Lev)']['WAR'].sum()

fig_bar, axes_bar = plt.subplots(2, 1, figsize=(14, 9.5))
categories_bar = ['Starting Pitchers', 'High-Leverage Relievers']
colors_bar = [sp_color, rp_color_v]
bar_h = 0.35
y_pos_bar = [0, bar_h * 1.4]

fig_bar.suptitle(f'Two Metrics, Two Stories\nHow WAR and WPA See High-Leverage Relievers ({START_YEAR}\u2013{END_YEAR})',
                 fontsize=20, fontweight='bold', fontname='Georgia', color='#333333', y=1.02)

def draw_white_gridlines(ax):
    ticks = ax.xaxis.get_major_locator().tick_values(*ax.get_xlim())
    for t in ticks:
        if t > 0:
            ax.axvline(t, color='white', linewidth=0.8, zorder=10)

# WPA subplot
ax_wpa = axes_bar[0]
wpa_values = [sp_total_wpa, rp_total_wpa]
bars_wpa = ax_wpa.barh(y_pos_bar, wpa_values, color=colors_bar, alpha=0.85, height=bar_h, zorder=2)
ax_wpa.set_yticks(y_pos_bar)
ax_wpa.set_yticklabels(categories_bar, fontsize=12, fontname='Georgia')
for b, val in zip(bars_wpa, wpa_values):
    ax_wpa.text(b.get_width() + 8, b.get_y() + b.get_height()/2,
                f'{val:.1f}\nWins Added', va='center', ha='left', fontsize=13, fontname='serif', linespacing=1.2)
ax_wpa.set_title('Win Probability Added (WPA)', fontsize=15, fontname='Georgia', color='#444', pad=2)
ax_wpa.xaxis.set_major_locator(mticker.AutoLocator())
ax_wpa.yaxis.grid(False); ax_wpa.xaxis.grid(False)
ax_wpa.tick_params(axis='x', length=0); ax_wpa.tick_params(axis='y', length=0)
for spine in ax_wpa.spines.values(): spine.set_visible(False)
ax_wpa.set_xlim(0, max(wpa_values) * 1.45)
ax_wpa.set_ylim(-bar_h * 0.6, y_pos_bar[-1] + bar_h * 0.9)
draw_white_gridlines(ax_wpa)

# WAR subplot
ax_war = axes_bar[1]
war_values = [sp_total_war, rp_total_war]
bars_war = ax_war.barh(y_pos_bar, war_values, color=colors_bar, alpha=0.85, height=bar_h, zorder=2)
ax_war.set_yticks(y_pos_bar)
ax_war.set_yticklabels(categories_bar, fontsize=12, fontname='Georgia')
for b, val in zip(bars_war, war_values):
    ax_war.text(b.get_width() + 40, b.get_y() + b.get_height()/2,
                f'{val:.1f}\nWins Above Replacement', va='center', ha='left', fontsize=13, fontname='serif', linespacing=1.2)
ax_war.set_title('Wins Above Replacement (WAR)', fontsize=15, fontname='Georgia', color='#444', pad=2)
ax_war.xaxis.set_major_locator(mticker.AutoLocator())
ax_war.yaxis.grid(False); ax_war.xaxis.grid(False)
ax_war.tick_params(axis='x', length=0); ax_war.tick_params(axis='y', length=0)
for spine in ax_war.spines.values(): spine.set_visible(False)
ax_war.set_xlim(0, max(war_values) * 1.55)
ax_war.set_ylim(-bar_h * 0.6, y_pos_bar[-1] + bar_h * 0.9)
draw_white_gridlines(ax_war)

plt.tight_layout(h_pad=10)
plt.subplots_adjust(top=0.90)
buf2 = io.BytesIO()
fig_bar.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_bar)
buf2.seek(0)
chart2_b64 = base64.b64encode(buf2.read()).decode('utf-8')
chart2_html = f'<img src="data:image/png;base64,{chart2_b64}" alt="WPA vs WAR Bar Chart" style="width:100%;max-width:900px;display:block;margin:0 auto;">'

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

# ── Top 50 WPA table (combined) ────────────────────────────────────────
cols_t = ['Player', 'WPA', 'WAR', 'IP', 'LI', 'Seasons', 'Pitcher_Type']
combined_top = pd.concat([
    sp_df[cols_t].assign(Role='SP'),
    high_lev_rp_df[cols_t].assign(Role='RP')
])
top50 = combined_top.nlargest(50, 'WPA').copy()
top50['WPA'] = top50['WPA'].round(1)
top50['WAR'] = top50['WAR'].round(1)
top50['IP']  = top50['IP'].round(1)
top50['LI']  = top50['LI'].round(2)
top50 = top50.reset_index(drop=True)
top50.index = top50.index + 1

table_rows = ''
for i, row in top50.iterrows():
    role = row['Role']
    cls = ' class="rp-row"' if role == 'RP' else ''
    table_rows += f'<tr{cls}><td>{i}</td><td>{row["Player"]}</td><td>{role}</td><td>{row["WPA"]:.1f}</td><td>{row["WAR"]:.1f}</td><td>{row["IP"]:.1f}</td><td>{row["LI"]:.2f}</td><td>{int(row["Seasons"])}</td></tr>\n'

table_combined_html = f'''<div class="top20-table" style="max-width:700px;margin:0 auto;">
  <table>
    <thead><tr><th>#</th><th>Player</th><th>Role</th><th>WPA</th><th>WAR</th><th>IP</th><th>pLI</th><th>Seasons</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>'''

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
<title>The Leverage Paradox: Rethinking the Value of Elite Relievers</title>
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
    background: linear-gradient(135deg, rgba(26,26,46,0.75) 0%, rgba(22,33,62,0.75) 50%, rgba(15,52,96,0.75) 100%),
               url('hero-stadium.jpg') center/cover no-repeat;
    color: #f0f0f0;
    padding: 6rem 2rem 5rem;
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
    font-size: 1.3rem;
    color: #d4d4d4;
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
  .cluster-tag.high {{ background: #D64541; }}
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
    background: #f4f7fa;
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    color: var(--text);
    padding: 1.8rem 2rem;
    border-radius: 0 8px 8px 0;
    margin-top: 1.5rem;
    font-size: 1.05rem;
    line-height: 1.7;
    font-style: italic;
  }}
  .takeaway strong {{
    font-style: normal;
  }}

  /* ── Top 50 Table ─────────────────────────────── */
  .top20-table {{
    overflow-x: auto;
    overflow-y: auto;
    max-height: 600px;
    -webkit-overflow-scrolling: touch;
  }}
  .top20-table table {{
    width: 100%;
    min-width: 520px;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}
  .top20-table th {{
    background: #f3f3f3;
    padding: 0.4rem 0.6rem;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 2;
  }}
  .top20-table td {{
    padding: 0.35rem 0.6rem;
    border-bottom: 1px solid #eee;
  }}
  .top20-table tr:hover td {{
    background: #fafafa;
  }}
  .top20-table tr.rp-row td {{
    background: #FFF3E6;
  }}
  .top20-table tr.rp-row:hover td {{
    background: #FFE8CC;
  }}

  /* ── Footer ───────────────────────────────────── */
  .method-details {{
    margin-top: 1.5rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .method-details summary {{
    padding: 0.8rem 1.2rem;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    color: var(--muted);
    background: #f9f9f9;
    list-style: none;
  }}
  .method-details summary::-webkit-details-marker {{ display: none; }}
  .method-details summary::before {{
    content: '\\25B6';
    display: inline-block;
    margin-right: 0.6rem;
    font-size: 0.7rem;
    transition: transform 0.2s;
  }}
  .method-details[open] summary::before {{
    transform: rotate(90deg);
  }}
  .method-details[open] summary {{
    border-bottom: 1px solid var(--border);
  }}
  .method-details > :not(summary) {{
    padding: 0 1.2rem;
  }}
  .method-details > .metrics {{
    padding: 0 1.2rem;
    margin-top: 1rem;
  }}
  .method-details > ul {{
    padding-left: 2.4rem;
    padding-right: 1.2rem;
    padding-bottom: 1.2rem;
  }}

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
    .hero {{ padding: 4rem 1.5rem 3.5rem; }}
    .container {{ padding: 0 1.2rem; }}
    .stat .number {{ font-size: 1.7rem; }}
    h2 {{ font-size: 1.5rem; }}
    .top20-table table {{ font-size: 0.82rem; }}
  }}
</style>
</head>
<body>

<!-- ── Hero ──────────────────────────────────────────────────────── -->
<header class="hero">
  <div class="container">
    <h1>The Leverage Paradox:<br>Rethinking the Value of Elite Relievers</h1>
    <p class="subtitle">
      An analysis of {n_total:,} pitcher-seasons from the past decade
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

    <div class="callout">
      <p><strong>The Core Tension:</strong> WAR says high-leverage relievers are barely worth noticing. WPA says they&rsquo;re the most impactful pitchers in baseball. How should this be reconciled?</p>
    </div>
  </div>
</section>

<!-- ── Data & Methodology ───────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>Data &amp; Methodology</h2>

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

    <details class="method-details">
      <summary>Classification, data sources &amp; notes</summary>

      <h3 style="margin-top:1.2rem;">Pitcher Classification</h3>
      <div class="metrics">
        <div class="metric-card">
          <strong>Starting Pitcher (SP)</strong>
          <p>Games Started &ge; 5 <em>and</em> Innings Pitched &ge; 20 in a season.</p>
        </div>
        <div class="metric-card">
          <strong>Relief Pitcher (RP)</strong>
          <p>Games &ge; 5 <em>and</em> Games Started &lt; 3. Pitchers meeting neither definition are excluded.</p>
        </div>
      </div>

      <ul style="margin-top:1.5rem;color:var(--muted);font-size:0.92rem;padding-left:1.2rem;">
        <li>All WAR values are <strong>fWAR</strong> (Fangraphs WAR), which uses FIP&mdash;strikeouts, walks, HBP, and home runs&mdash;rather than runs allowed, isolating repeatable pitching skill from fielding and sequencing.</li>
        <li style="margin-top:0.4rem;">Scope: 10 MLB seasons [{START_YEAR}&ndash;{END_YEAR}]</li>
        <li style="margin-top:0.4rem;">Minimum {QUAL} IP per season</li>
        <li style="margin-top:0.4rem;">All data from <a href="https://www.fangraphs.com/" style="color: var(--accent);">Fangraphs</a> via <a href="https://github.com/jldbc/pybaseball" style="color: var(--accent);">pybaseball</a> &mdash; see the <a href="https://blogs.fangraphs.com/glossary/" style="color: var(--accent);">Fangraphs Glossary</a> for stat definitions</li>
      </ul>
    </details>
  </div>
</section>

<!-- ── SP/RP Divide ──────────────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>Volume and Leverage</h2>
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

    <div class="metrics" style="margin-top:1.5rem;">
      <div class="metric-card">
        <strong>SPs Eat Innings</strong>
        <p>SPs dominate the workload &mdash; a typical SP season covers 3&ndash;4&times; the innings of a typical RP season, which is precisely why WAR favors them so heavily.</p>
      </div>
      <div class="metric-card">
        <strong>Leverage Concentration</strong>
        <p>SPs cluster around league-average pLI (~1.0), but RPs show enormous variance. A substantial group consistently operates well above that baseline &mdash; the high-leverage arms deployed in the highest-stakes situations.</p>
      </div>
    </div>
  </div>
</section>

<!-- ── Clustering ────────────────────────────────────────────────── -->
<section>
  <div class="container">
    <h2>What Constitutes a High-Leverage Reliever?</h2>
    <h3 style="margin-top:0.2rem;">The Giovanni Gallegos Line</h3>
    <p>
      To move beyond per-season snapshots, we aggregate each pitcher&rsquo;s career totals across
      all seasons in the dataset, then apply k-means clustering (k&nbsp;=&nbsp;3) on career-average pLI
      to separate relievers into three distinct tiers. Because pLI captures how often a pitcher
      is deployed in high-stakes situations, this clustering effectively identifies the
      subset of relievers whose managers trust them with the game on the line &mdash;
      the closers, setup men, and firemen who anchor a bullpen.
      Under this framework, the cutoff between mid- and high-leverage sits at a pLI of 1.08,
      corresponding to the 10-year average leverage of pitchers like Giovanny Gallegos,
      Tyler Ferguson, and Joe Jim&eacute;nez.
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

<!-- ── Two Metrics, Two Stories ───────────────────────────────────── -->
<section>
  <div class="container">
    <h2>WAR, WPA and the Elusive Value of Systemic Leverage</h2>
    <p>
      At their core, WAR and WPA answer different questions. WAR asks how much total value
      a player provides over a season, independent of context, while WPA asks who actually
      swings games in real time. The divergence between these lenses is nowhere more apparent
      than in the evaluation of elite relievers.
    </p>
    <p style="font-style:italic;color:var(--muted);font-size:0.93rem;margin-top:0.3rem;">
      WAR and WPA both measure value, but they rest on different baselines, making
      direct scalar comparison invalid. Still, contrasting the stories they tell
      reveals important structural insights.
    </p>

    <div class="callout">
      <p>Consider the asymmetry: Zach Wheeler throws six strong innings in a comfortable lead
      and shifts win probability by only a few percentage points. Jhoan Duran enters
      in the eighth with the tying run on third, strikes out two batters, and swings win probability by
      30&ndash;40% in two minutes. Both performances require real skill &mdash; but WPA rewards
      Duran&rsquo;s two minutes far more than Wheeler&rsquo;s six innings, not because
      Duran is better, but because the <em>situation</em> carried more weight.</p>
    </div>

    <h3>Season-Level View</h3>
    <p>
      Each dot below is a single pitcher-season. SPs spread horizontally (high WAR, moderate WPA) &mdash;
      their volume drives skill-based value but dilutes game-context impact.
      High-leverage RPs spread vertically (low WAR, wide WPA range) &mdash;
      their value is concentrated in high-stakes moments but inflated by
      sequencing and managerial deployment.
      The gap between the two clusters is the valuation no-man&rsquo;s-land where
      neither metric tells the full story.
    </p>

    <div class="chart-wrap">{chart3_html}</div>

    <h3>Cumulative Value: {START_YEAR}&ndash;{END_YEAR}</h3>
    <p>
      Zooming out from individual seasons to aggregate totals over the full decade
      sharpens the contrast. The bar chart and leaderboard below show cumulative
      WAR and WPA for all SPs vs. the high-leverage reliever tier identified above.
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

    <div style="background:#f4f7fa;border-left:4px solid var(--accent);border-radius:6px;padding:1rem 1.4rem;margin:1.5rem auto 0;max-width:620px;text-align:center;">
      <p style="font-style:italic;color:#444;font-size:0.97rem;margin:0;">
        By WPA, high-leverage relievers outproduce all starting pitchers combined.
        By WAR, they&rsquo;re worth a fraction.
      </p>
    </div>

    <h3>Top 50 Pitchers by Aggregate WPA</h3>
    <p>
      A single leaderboard combining SPs and high-leverage RPs, sorted by total WPA.
      It&rsquo;s telling that relievers appear so prominently on this list &mdash; alongside
      aces who threw roughly five times as many innings. This reinforces that elite
      high-leverage performance of individual relievers persists well beyond short-term variance.
    </p>
    {table_combined_html}
  </div>
</section>

<!-- ── Conclusion ────────────────────────────────────────────────── -->
<section class="conclusion">
  <div class="container">
    <h2>Conclusion</h2>
    <ul>
      <li>
        <strong>WAR</strong> is designed to credit total volume of performance and isolate underlying skill,
        by treating all innings as equally important and ignoring leverage. This leads to systematic
        undervaluation of relievers.
      </li>
      <li>
        <strong>WPA</strong> measures who actually swings game outcomes, fully incorporating timing and context,
        but in doing so systematically discredits the baseline win probability provided by quality starts
        and in turn overvalues relievers by absorbing managerial usage and sequencing factors.
      </li>
    </ul>

    <p style="margin-top:1rem;color:var(--muted);">
      Yet if these contextual factors were purely noise, their effects would largely cancel out over
      large samples. Instead, over a decade of league-wide data, high-leverage relievers consistently
      dominate aggregate WPA, reflecting a durable structural feature of the game: leverage is
      systematically concentrated into a small number of bullpen innings. This is made explicit in pLI
      data, where elite relievers operate at roughly double the average leverage of starters, placing
      them in the most consequential moments of nearly every game.
    </p>

    <div class="takeaway">
      Starters and offenses establish win probability baseline; high-leverage relievers resolve it.
      By controlling the final allocation of wins, elite relievers occupy a structurally distinct
      role that makes their true value meaningfully larger than WAR suggests, even if the precise
      magnitude remains debatable.
    </div>
  </div>
</section>

<footer>
  <div class="container">
    Data: Fangraphs via pybaseball &middot; {START_YEAR}&ndash;{END_YEAR} &middot; Min {QUAL} IP
    <br>Hero photo by <a href="https://www.pexels.com/@shawnreza/" style="color:#bbb;">Shawn Reza</a> on <a href="https://www.pexels.com/" style="color:#bbb;">Pexels</a>
  </div>
</footer>

</body>
</html>
"""

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Built {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)')
