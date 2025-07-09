# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from scipy.optimize import curve_fit
from IPython.display import display
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')
import os
!pip install xlsxwriter

# --- CONFIGURATION ---
REPEATABLE = True
if REPEATABLE:
    random.seed(42)
    np.random.seed(42)

grid_size = 100
x_vals = np.linspace(0, 10, grid_size)
y_vals = np.linspace(0, 10, grid_size)
X, Y = np.meshgrid(x_vals, y_vals)

num_sensors = 10
timesteps = 10
sensor_range = 2.5
monte_carlo_runs = 50

# --- SENSOR GENERATION ---
def generate_sensor(i):
    return {
        'id': f'S{i+1}',
        'pos': [random.uniform(1, 9), random.uniform(1, 9)],
        'range': sensor_range,
        'noise': random.uniform(0.05, 0.2),
        'battery': random.uniform(0.6, 1.0),
        'drift': random.uniform(0.01, 0.1),
        'fail_risk': random.uniform(0.01, 0.2)
    }

# --- NEUTROSOPHIC FUNCTIONS ---
def compute_neutrosophic_layers(sensor, X, Y):
    sx, sy = sensor['pos']
    r = sensor['range']
    distance = np.sqrt((X - sx)**2 + (Y - sy)**2)
    coverage = np.clip(1 - distance / r, 0, 1)

    noise = sensor['noise']
    battery = sensor['battery']
    drift = sensor['drift']
    fail_risk = sensor['fail_risk']

    mu = coverage * (1 - noise) * battery
    sigma = (noise + drift + (1 - battery)) * coverage
    nu = ((1 - coverage) * (1 - battery) + fail_risk) * (1 - coverage)
    return mu, sigma, nu, coverage

def neutrosophic_interior(mu, threshold=0.6):
    return mu > threshold

def neutrosophic_exterior(nu, threshold=0.6):
    return nu > threshold

def neutrosophic_boundary(sigma, threshold=0.4):
    return sigma > threshold

def neutrosophic_closure(mu, sigma, threshold_mu=0.4, threshold_sigma=0.2):
    return (mu > threshold_mu) | (sigma > threshold_sigma)

def neutrosophic_subspace(mask):
    return np.where(mask)

# --- INITIAL SIMULATION (1 RUN) ---
sensors = [generate_sensor(i) for i in range(num_sensors)]
results = []

for t in range(timesteps):
    mu_total = np.zeros_like(X)
    sigma_total = np.zeros_like(X)
    nu_total = np.zeros_like(X)
    trust_scores = []

    for sensor in sensors:
        mu_layer, sigma_layer, nu_layer, coverage = compute_neutrosophic_layers(sensor, X, Y)
        mu_total += mu_layer
        sigma_total += sigma_layer
        nu_total += nu_layer

        mask = coverage > 0
        if np.any(mask):
            mu_vals = mu_layer[mask]
            sigma_vals = sigma_layer[mask]
            nu_vals = nu_layer[mask]
            denom = mu_vals + sigma_vals + nu_vals
            norm_trust_vals = np.divide(mu_vals, denom, out=np.zeros_like(mu_vals), where=denom != 0)
            trust_score = np.mean(norm_trust_vals)
        else:
            trust_score = 0

        trust_scores.append({
            'Sensor': sensor['id'],
            'Trust Score': round(trust_score, 4),
            'Battery': round(sensor['battery'], 2),
            'Noise': round(sensor['noise'], 2),
            'Drift': round(sensor['drift'], 2),
            'Fail Risk': round(sensor['fail_risk'], 2)
        })

        sensor['battery'] = max(sensor['battery'] - 0.05, 0)
        sensor['drift'] += 0.005
        sensor['fail_risk'] = min(sensor['fail_risk'] + 0.01, 1.0)

    trust_df = pd.DataFrame(trust_scores).sort_values(by='Trust Score', ascending=False)
    results.append((t, trust_df))

    print(f"\n=== Time Step {t} ===")
    display(trust_df)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].imshow(mu_total, extent=(0, 10, 0, 10), origin='lower', cmap='YlGnBu')
    axs[0].set_title(f"μ - Membership (t={t})")
    axs[1].imshow(sigma_total, extent=(0, 10, 0, 10), origin='lower', cmap='YlOrBr')
    axs[1].set_title(f"σ - Indeterminacy (t={t})")
    axs[2].imshow(nu_total, extent=(0, 10, 0, 10), origin='lower', cmap='Purples')
    axs[2].set_title(f"ν - Non-membership (t={t})")
    for ax in axs:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    plt.tight_layout()
    plt.show()

# --- MONTE CARLO SIMULATION ---
all_runs_scores = []
ranking_frequency = {f"S{i+1}": 0 for i in range(num_sensors)}
final_trust_collection = {f"S{i+1}": [] for i in range(num_sensors)}

for run in range(monte_carlo_runs):
    sensors = [generate_sensor(i) for i in range(num_sensors)]
    sensor_trust_timeline = {sensor['id']: [] for sensor in sensors}

    for t in range(timesteps):
        for sensor in sensors:
            mu_layer, sigma_layer, nu_layer, coverage = compute_neutrosophic_layers(sensor, X, Y)
            mask = coverage > 0
            if np.any(mask):
                mu_vals = mu_layer[mask]
                sigma_vals = sigma_layer[mask]
                nu_vals = nu_layer[mask]
                denom = mu_vals + sigma_vals + nu_vals
                norm_trust_vals = np.divide(mu_vals, denom, out=np.zeros_like(mu_vals), where=denom != 0)
                trust_score = np.mean(norm_trust_vals)
            else:
                trust_score = 0

            sensor_trust_timeline[sensor['id']].append(trust_score)

            sensor['battery'] = max(sensor['battery'] - 0.05, 0)
            sensor['drift'] += 0.005
            sensor['fail_risk'] = min(sensor['fail_risk'] + 0.01, 1.0)

        if t == timesteps - 1:
            final_scores = [(sensor_id, sensor_trust_timeline[sensor_id][-1]) for sensor_id in sensor_trust_timeline]
            for sid, score in final_scores:
                final_trust_collection[sid].append(score)
            sorted_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
            for rank, (sensor_id, _) in enumerate(sorted_scores[:3]):
                ranking_frequency[sensor_id] += 1

    run_df = pd.DataFrame(sensor_trust_timeline)
    run_df['Timestep'] = list(range(timesteps))
    run_df['Run'] = run
    all_runs_scores.append(run_df)

# --- AGGREGATE RESULTS ---
combined_df = pd.concat(all_runs_scores)
mean_trust = combined_df.groupby('Timestep').mean()
std_trust = combined_df.groupby('Timestep').std()

plt.figure(figsize=(12, 6))
for sensor_id in mean_trust.columns:
    plt.plot(mean_trust.index, mean_trust[sensor_id], label=sensor_id)
    plt.fill_between(mean_trust.index,
                     mean_trust[sensor_id] - std_trust[sensor_id],
                     mean_trust[sensor_id] + std_trust[sensor_id],
                     alpha=0.2)
plt.title("Average Trust Score Over Time with Confidence Bands")
plt.xlabel("Timestep")
plt.ylabel("Normalized Trust Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

ranking_df = pd.DataFrame.from_dict(ranking_frequency, orient='index', columns=['Top-3 Appearances'])
ranking_df['Top-3 %'] = (ranking_df['Top-3 Appearances'] / monte_carlo_runs) * 100
ranking_df = ranking_df.sort_values(by='Top-3 %', ascending=False)

summary_stats = []
for sid, scores in final_trust_collection.items():
    summary_stats.append({
        "Sensor": sid,
        "Avg. Final Trust Score": np.mean(scores),
        "Trust StdDev": np.std(scores)
    })
summary_df = pd.DataFrame(summary_stats).sort_values(by="Avg. Final Trust Score", ascending=False)

print("\n📊 Ranking Frequency (% Top-3 appearances):")
display(ranking_df)
print("\n📉 Final Trust Score Summary (Average and Std Dev):")
display(summary_df)

mean_trust.to_csv("montecarlo_mean_trust.csv")
ranking_df.to_csv("montecarlo_ranking_frequency.csv")
summary_df.to_csv("montecarlo_final_stats.csv")

with pd.ExcelWriter("montecarlo_summary.xlsx", engine='xlsxwriter') as writer:
    mean_trust.to_excel(writer, sheet_name='Mean Trust')
    std_trust.to_excel(writer, sheet_name='Trust StdDev')
    ranking_df.to_excel(writer, sheet_name='Ranking Frequency')
    summary_df.to_excel(writer, sheet_name='Final Trust Stats')

print("\n✅ Exported:")
print(" - montecarlo_mean_trust.csv")
print(" - montecarlo_ranking_frequency.csv")
print(" - montecarlo_final_stats.csv")
print(" - montecarlo_summary.xlsx")

# --- ANIMATED HEATMAP BLOCK ---
sensors_anim = [generate_sensor(i) for i in range(num_sensors)]
mu_frames, sigma_frames, nu_frames = [], [], []

for t in range(timesteps):
    mu_total = np.zeros_like(X)
    sigma_total = np.zeros_like(X)
    nu_total = np.zeros_like(X)

    for sensor in sensors_anim:
        mu_layer, sigma_layer, nu_layer, _ = compute_neutrosophic_layers(sensor, X, Y)
        mu_total += mu_layer
        sigma_total += sigma_layer
        nu_total += nu_layer

        sensor['battery'] = max(sensor['battery'] - 0.05, 0)
        sensor['drift'] += 0.005
        sensor['fail_risk'] = min(sensor['fail_risk'] + 0.01, 1.0)

    mu_frames.append(mu_total.copy())
    sigma_frames.append(sigma_total.copy())
    nu_frames.append(nu_total.copy())

    interior = neutrosophic_interior(mu_total)
    boundary = neutrosophic_boundary(sigma_total)
    exterior = neutrosophic_exterior(nu_total)

os.makedirs("animations", exist_ok=True)

def create_animation(frames, title, filename, cmap):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames[0], extent=(0, 10, 0, 10), origin='lower', cmap=cmap)
    ax.set_title(f"{title} (t=0)")
    def update(frame):
        im.set_array(frames[frame])
        ax.set_title(f"{title} (t={frame})")
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(f"animations/{filename}.gif", writer='pillow', fps=2)
    plt.close(fig)

create_animation(mu_frames, "Membership (μ)", "mu_animation", 'YlGnBu')
create_animation(sigma_frames, "Indeterminacy (σ)", "sigma_animation", 'YlOrBr')
create_animation(nu_frames, "Non-membership (ν)", "nu_animation", 'Purples')

print("\n Animated heatmaps saved to animations/*.gif")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from scipy.optimize import curve_fit
from IPython.display import display
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')
import os
!pip install xlsxwriter

# --- CONFIGURATION ---
REPEATABLE = True
if REPEATABLE:
    random.seed(42)
    np.random.seed(42)

grid_size = 100
x_vals = np.linspace(0, 10, grid_size)
y_vals = np.linspace(0, 10, grid_size)
X, Y = np.meshgrid(x_vals, y_vals)

num_sensors = 10
timesteps = 10
sensor_range = 2.5
monte_carlo_runs = 50

# --- SENSOR GENERATION ---
def generate_sensor(i):
    return {
        'id': f'S{i+1}',
        'pos': [random.uniform(1, 9), random.uniform(1, 9)],
        'range': sensor_range,
        'noise': random.uniform(0.05, 0.2),
        'battery': random.uniform(0.6, 1.0),
        'drift': random.uniform(0.01, 0.1),
        'fail_risk': random.uniform(0.01, 0.2)
    }

# --- NEUTROSOPHIC FUNCTIONS ---
def compute_neutrosophic_layers(sensor, X, Y):
    sx, sy = sensor['pos']
    r = sensor['range']
    distance = np.sqrt((X - sx)**2 + (Y - sy)**2)
    coverage = np.clip(1 - distance / r, 0, 1)

    noise = sensor['noise']
    battery = sensor['battery']
    drift = sensor['drift']
    fail_risk = sensor['fail_risk']

    mu = coverage * (1 - noise) * battery
    sigma = (noise + drift + (1 - battery)) * coverage
    nu = ((1 - coverage) * (1 - battery) + fail_risk) * (1 - coverage)
    return mu, sigma, nu, coverage

def neutrosophic_interior(mu, threshold=0.6):
    return mu > threshold

def neutrosophic_exterior(nu, threshold=0.6):
    return nu > threshold

def neutrosophic_boundary(sigma, threshold=0.4):
    return sigma > threshold

def neutrosophic_closure(mu, sigma, threshold_mu=0.4, threshold_sigma=0.2):
    return (mu > threshold_mu) | (sigma > threshold_sigma)

def neutrosophic_subspace(mask):
    return np.where(mask)

# --- ANIMATED HEATMAP BLOCK ---
sensors_anim = [generate_sensor(i) for i in range(num_sensors)]
mu_frames, sigma_frames, nu_frames = [], [], []
interior_areas, boundary_areas, exterior_areas = [], [], []

for t in range(timesteps):
    mu_total = np.zeros_like(X)
    sigma_total = np.zeros_like(X)
    nu_total = np.zeros_like(X)

    for sensor in sensors_anim:
        mu_layer, sigma_layer, nu_layer, _ = compute_neutrosophic_layers(sensor, X, Y)
        mu_total += mu_layer
        sigma_total += sigma_layer
        nu_total += nu_layer

        sensor['battery'] = max(sensor['battery'] - 0.05, 0)
        sensor['drift'] += 0.005
        sensor['fail_risk'] = min(sensor['fail_risk'] + 0.01, 1.0)

    mu_frames.append(mu_total.copy())
    sigma_frames.append(sigma_total.copy())
    nu_frames.append(nu_total.copy())

    interior = neutrosophic_interior(mu_total)
    boundary = neutrosophic_boundary(sigma_total)
    exterior = neutrosophic_exterior(nu_total)

    # Save area coverage (sum of true values)
    interior_areas.append(np.sum(interior))
    boundary_areas.append(np.sum(boundary))
    exterior_areas.append(np.sum(exterior))

    # Plot masks
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(interior, extent=(0, 10, 0, 10), origin='lower', cmap='Greens')
    axs[0].set_title(f"Interior Mask (t={t})")

    axs[1].imshow(boundary, extent=(0, 10, 0, 10), origin='lower', cmap='Oranges')
    axs[1].set_title(f"Boundary Mask (t={t})")

    axs[2].imshow(exterior, extent=(0, 10, 0, 10), origin='lower', cmap='Reds')
    axs[2].set_title(f"Exterior Mask (t={t})")

    for ax in axs:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(f"animations/neutrosophic_masks_t{t}.png")
    plt.close()

# Export coverage area stats
coverage_df = pd.DataFrame({
    'Timestep': list(range(timesteps)),
    'Interior Area': interior_areas,
    'Boundary Area': boundary_areas,
    'Exterior Area': exterior_areas
})

coverage_df.to_csv("neutrosophic_region_areas.csv", index=False)

# Create directory if not exists
os.makedirs("animations", exist_ok=True)

def create_animation(frames, title, filename, cmap):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames[0], extent=(0, 10, 0, 10), origin='lower', cmap=cmap)
    ax.set_title(f"{title} (t=0)")

    def update(frame):
        im.set_array(frames[frame])
        ax.set_title(f"{title} (t={frame})")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(f"animations/{filename}.gif", writer='pillow', fps=2)
    plt.close(fig)

create_animation(mu_frames, "Membership (μ)", "mu_animation", 'YlGnBu')
create_animation(sigma_frames, "Indeterminacy (σ)", "sigma_animation", 'YlOrBr')
create_animation(nu_frames, "Non-membership (ν)", "nu_animation", 'Purples')

print("\n Animated heatmaps saved to animations/*.gif")
print("Neutrosophic region coverage stats saved to neutrosophic_region_areas.csv")

# --- Combined Plot for All Masks Across Time ---
fig, axs = plt.subplots(timesteps, 3, figsize=(15, 3 * timesteps))

for t in range(timesteps):
    mu_total = mu_frames[t]
    sigma_total = sigma_frames[t]
    nu_total = nu_frames[t]

    interior = neutrosophic_interior(mu_total)
    boundary = neutrosophic_boundary(sigma_total)
    exterior = neutrosophic_exterior(nu_total)

    axs[t, 0].imshow(interior, extent=(0, 10, 0, 10), origin='lower', cmap='Greens')
    axs[t, 0].set_title(f"Interior (t={t})")

    axs[t, 1].imshow(boundary, extent=(0, 10, 0, 10), origin='lower', cmap='Oranges')
    axs[t, 1].set_title(f"Boundary (t={t})")

    axs[t, 2].imshow(exterior, extent=(0, 10, 0, 10), origin='lower', cmap='Reds')
    axs[t, 2].set_title(f"Exterior (t={t})")

    for j in range(3):
        axs[t, j].set_xticks([])
        axs[t, j].set_yticks([])

plt.tight_layout()
plt.savefig("animations/all_neutrosophic_masks_grid.png")
plt.close()

print(" Grid of all neutrosophic masks saved as animations/all_neutrosophic_masks_grid.png")

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# Ensure the output directory exists
os.makedirs("animations", exist_ok=True)

# List to store paths of individual frames
frame_paths = []

for t in range(timesteps):
    mu_total = mu_frames[t]
    sigma_total = sigma_frames[t]
    nu_total = nu_frames[t]

    interior = neutrosophic_interior(mu_total)
    boundary = neutrosophic_boundary(sigma_total)
    exterior = neutrosophic_exterior(nu_total)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(interior, extent=(0, 10, 0, 10), origin='lower', cmap='Greens')
    axs[0].set_title("Interior")

    axs[1].imshow(boundary, extent=(0, 10, 0, 10), origin='lower', cmap='Oranges')
    axs[1].set_title("Boundary")

    axs[2].imshow(exterior, extent=(0, 10, 0, 10), origin='lower', cmap='Reds')
    axs[2].set_title("Exterior")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Neutrosophic Masks at t={t}")
    frame_path = f"animations/frame_{t}.png"
    plt.savefig(frame_path)
    frame_paths.append(frame_path)
    plt.close()

# Create animated GIF
gif_path = "animations/neutrosophic_masks.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5, loop=0) as writer:
    for path in frame_paths:
        image = imageio.imread(path)
        writer.append_data(image)

print(f" GIF created at: {gif_path}")
