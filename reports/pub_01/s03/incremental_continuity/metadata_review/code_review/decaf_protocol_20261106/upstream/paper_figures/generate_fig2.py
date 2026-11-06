import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Dictionary mapping model names to their test_results.csv file paths
MODEL_CSV_PATHS = {
    # Baseline Models
    'BaselineDecoder2': '../checkpoints_baseline_models/BaselineDecoder2_pearson_l1/subject_independent/baseline_decoder2_20250909_020422/best_model_test_recursive_results_20250909_153615/test_results_recursive.csv',
    'Happyquoka2': '../checkpoints_baseline_models/Happyquoka2_pearson_l1/subject_independent/happyquoka2_20250908_221332/best_model_test_recursive_results_20250909_154021/test_results_recursive.csv',
    'Vlaai': '../checkpoints_baseline_models/Vlaai_pearson_l1/subject_independent/vlaai_20250905_181450/best_model_test_recursive_results_20250909_153822/test_results_recursive.csv',
    
    # Proposed Models
    'TwoBranch_Recursive_3.0sec': '../checkpoints_two_branch_dynamic_recursive_gru_teacherforcing_train/Old_TwoBranchRecursiveContext3.0SecDynamic_pearson_l1/subject_independent/two_branch_recursive_context_3.0sec_dynamic_20250910_104156/best_model_test_results_20250910_171331/test_results.csv',
    'TwoBranch_Recursive_3.0sec_GT': '../checkpoints_two_branch_dynamic_recursive_gru_teacherforcing_train/TwoBranchRecursiveContext3.0SecDynamic_pearson_l1/subject_independent/two_branch_recursive_context_3.0sec_dynamic_20250910_170516/best_model_test_results_teacher_forcing_20250910_192353/test_results.csv',
    
    # Two Branch Dynamic LSTM Models
    #'TwoBranch_3.0sec': '../checkpoints_two_branch_dynamic_recursive_gru_teacherforcing_train/TwoBranchRecursiveContext2.0SecDynamic_pearson_l1/subject_independent/two_branch_recursive_context_2.0sec_dynamic_20250910_133832/best_model_test_results_teacher_forcing_20250910_165227/test_results.csv'
    }

    # Default color palette for models
DEFAULT_MODEL_COLORS = {
    # Baseline Models - All same color (gray)
    'BaselineDecoder2': '#f5f5f5',   # Gray
    'Happyquoka2': '#f5f5f5',        # Gray
    'Vlaai': '#f5f5f5',              # Gray
    
    # Proposed Models - Distinct colors
    'TwoBranch_Recursive_3.0sec': '#E1D5E7',  # Light Purple
    'TwoBranch_Recursive_3.0sec_GT': '#9467bd',  # Darker Purple for Oracle
    
}


def extract_subject_scores(csv_path: str) -> Dict[str, float]:
    """
    Extract subject-wise mean scores from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing test results
    
    Returns:
        Dict[str, float]: Dictionary with subject ID as key and mean correlation score as value
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV is empty or contains invalid data
    """
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Remove any rows with NaN values in the required columns
        df_clean = df.dropna(subset=['sub', 'correlation'])
        
        if df_clean.empty:
            raise ValueError("No valid data found after removing NaN values")
        
        # Group by subject and calculate mean correlation scores
        subject_means = df_clean.groupby('sub')['correlation'].mean()
        
        # Convert to dictionary with string keys and float values
        subject_scores_dict = {str(subject_id): float(score) for subject_id, score in subject_means.items()}
        
        return subject_scores_dict
        
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty or corrupted")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")


def plot_model_violin_comparison(model_colors: Optional[Dict[str, str]] = None, 
                                figsize: tuple = (12, 6),
                                save_path: Optional[str] = None,
                                show_plot: bool = True,
                                dpi: int = 300) -> plt.Figure:
    """
    Create violin plots comparing all models' subject-wise correlation scores.
    
    Args:
        model_colors (Dict[str, str], optional): Dictionary mapping model names to hex colors.
                                               If None, uses DEFAULT_MODEL_COLORS.
        figsize (tuple): Figure size (width, height). Default: (14, 8)
        save_path (str, optional): Path to save the figure. If None, doesn't save.
        show_plot (bool): Whether to display the plot. Default: True
    
    Returns:
        plt.Figure: The matplotlib figure object
    """
    
    # Use default colors if none provided
    if model_colors is None:
        model_colors = DEFAULT_MODEL_COLORS.copy()
    
    # Collect all data for plotting
    plot_data = []
    
    print("Loading data for all models...")
    for model_name, csv_path in MODEL_CSV_PATHS.items():
        try:
            subject_scores = extract_subject_scores(csv_path)
            
            # Add each subject's score to the plot data
            for subject_id, score in subject_scores.items():
                plot_data.append({
                    'Model': model_name,
                    'Subject': subject_id,
                    'Correlation': score
                })
            
            print(f"✓ Loaded {model_name}: {len(subject_scores)} subjects")
            
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        raise ValueError("No data could be loaded for plotting")
    
    # Create the plot with publication-ready styling
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'grid.alpha': 0.3
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define custom order for x-axis - only include models with data
    preferred_order = [
        'BaselineDecoder2', 
        'Vlaai', 
        'Happyquoka2', 
        'TwoBranch_Recursive_3.0sec',
        'TwoBranch_Recursive_3.0sec_GT'
    ]
    
    # Filter to only include models that have data
    available_models = df['Model'].unique()
    model_order = [model for model in preferred_order if model in available_models]
    
    # Create cleaner display names
    display_names = {
        'BaselineDecoder2': 'Linear',
        'Vlaai': 'VLAAI', 
        'Happyquoka2': 'HappyQuoka',
        'TwoBranch_Recursive_3.0sec': 'DECAF',
        'TwoBranch_Recursive_3.0sec_GT': 'DECAF_Oracle'
    }
    
    # Create color list in the correct order
    colors = [model_colors.get(model, '#333333') for model in model_order]
    
    # Create violin plot
    parts = ax.violinplot([df[df['Model'] == model]['Correlation'].values for model in model_order],
                         positions=range(len(model_order)),
                         showmeans=True,
                         showmedians=True,
                         widths=0.7)
    
    # Customize colors and styling
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    
    # Customize other elements for publication
    parts['cmeans'].set_color('white')
    parts['cmeans'].set_linewidth(2.5)
    parts['cmeans'].set_alpha(1.0)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.8)
    parts['cmedians'].set_alpha(1.0)
    
    # Style the whiskers and caps
    for partname in ('cbars', 'cmins', 'cmaxes'):
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1.2)
    
    # Customize the plot for publication
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([display_names[model] for model in model_order], rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Correlation Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    
    # Remove title for cleaner look (can be added in caption)
    # ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Add subtle grid for better readability
    ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set y-axis limits for better visualization (add extra space at bottom for mean lines)
    y_min = df['Correlation'].min() - 0.08
    y_max = df['Correlation'].max() + 0.03
    ax.set_ylim(y_min, y_max)
    
    # Add mean value lines at the bottom for each model
    mean_line_y = y_min + 0.03  # Position for the mean lines
    for i, model in enumerate(model_order):
        model_data = df[df['Model'] == model]['Correlation']
        mean_val = model_data.mean()
        
        # Draw horizontal line showing the mean
        line_width = 0.4  # Width of the mean line
        ax.plot([i - line_width/2, i + line_width/2], [mean_line_y, mean_line_y], 
                color='black', linewidth=2.5, solid_capstyle='butt', zorder=10)
        
        # Add mean value text below the line
        ax.text(i, mean_line_y - 0.015, f'{mean_val:.3f}', 
                ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Add subtle separators between model groups
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)  # After baseline models
    
    # Add group labels
    ax.text(1, y_max - 0.01, 'Baseline Models', ha='center', va='top', 
            fontsize=10, style='italic', color='gray')
    ax.text(3.5, y_max - 0.01, 'Proposed Models', ha='center', va='top',
            fontsize=10, style='italic', color='gray')
    
    # Adjust layout with more space
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92)
    
    # Save if path provided with high quality
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='png')
        print(f"Figure saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def compare_happyquoka_vs_eeg_pred():
    """
    Statistical comparison between HappyQuoka and EEG-Pred models.
    
    Returns:
        dict: Statistical test results
    """
    
    print("Statistical Comparison: HappyQuoka vs EEG-Pred")
    print("=" * 50)
    
    # Load data for both modelsss
    try:
        happyquoka_scores = extract_subject_scores(MODEL_CSV_PATHS['Happyquoka2'])
        eeg_pred_scores = extract_subject_scores(MODEL_CSV_PATHS['TwoBranch_Recursive_3.0sec'])
        
        print(f"HappyQuoka subjects loaded: {len(happyquoka_scores)}")
        print(f"EEG-Pred subjects loaded: {len(eeg_pred_scores)}")
        
        # Ensure we have the same subjects
        common_subjects = set(happyquoka_scores.keys()) & set(eeg_pred_scores.keys())
        
        if len(common_subjects) != len(happyquoka_scores):
            print(f"Warning: Only {len(common_subjects)} subjects found in both models")
        
        # Extract paired scores
        happyquoka_values = [happyquoka_scores[subj] for subj in sorted(common_subjects)]
        eeg_pred_values = [eeg_pred_scores[subj] for subj in sorted(common_subjects)]
        
        # Calculate descriptive statistics
        hq_mean = np.mean(happyquoka_values)
        hq_std = np.std(happyquoka_values, ddof=1)
        ep_mean = np.mean(eeg_pred_values)
        ep_std = np.std(eeg_pred_values, ddof=1)
        
        print(f"\nDescriptive Statistics:")
        print(f"HappyQuoka:  Mean = {hq_mean:.4f}, SD = {hq_std:.4f}")
        print(f"EEG-Pred:    Mean = {ep_mean:.4f}, SD = {ep_std:.4f}")
        print(f"Difference:  {ep_mean - hq_mean:.4f} (EEG-Pred - HappyQuoka)")
        
        # Perform Wilcoxon signed-rank test (paired, non-parametric)
        statistic, p_value = stats.wilcoxon(happyquoka_values, eeg_pred_values, 
                                          alternative='two-sided')
        
        # Calculate effect size (r = Z / sqrt(N))
        n = len(common_subjects)
        z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else np.inf
        effect_size = abs(z_score) / np.sqrt(n) if np.isfinite(z_score) else np.nan
        
        # Interpret results
        print(f"\nWilcoxon Signed-Rank Test:")
        print(f"Statistic: {statistic}")
        print(f"P-value: {p_value:.6f}")
        print(f"Effect size (r): {effect_size:.4f}")
        
        # Interpretation
        if p_value < 0.001:
            significance = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        else:
            significance = "not significant (p ≥ 0.05)"
        
        print(f"\nResult: The difference is {significance}")
        
        if ep_mean > hq_mean:
            direction = "EEG-Pred performs better than HappyQuoka"
        else:
            direction = "HappyQuoka performs better than EEG-Pred"
        
        print(f"Direction: {direction}")
        
        # Effect size interpretation
        if effect_size < 0.1:
            effect_interp = "negligible"
        elif effect_size < 0.3:
            effect_interp = "small"
        elif effect_size < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        print(f"Effect size: {effect_interp}")
        
        return {
            'happyquoka_mean': hq_mean,
            'happyquoka_std': hq_std,
            'eeg_pred_mean': ep_mean,
            'eeg_pred_std': ep_std,
            'difference': ep_mean - hq_mean,
            'n_subjects': n,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significance': significance
        }
        
    except Exception as e:
        print(f"Error in statistical comparison: {e}")
        return None


def plot_subject_trajectories(model_colors: Optional[Dict[str, str]] = None,
                             figsize: tuple = (10, 6),
                          save_path: Optional[str] = None,
                          show_plot: bool = True,
                          dpi: int = 300) -> plt.Figure:
    """
    Create a plot showing individual subject trajectories across models.
    """
    
    if model_colors is None:
        model_colors = DEFAULT_MODEL_COLORS.copy()
    
    # Load all data
    plot_data = []
    for model_name, csv_path in MODEL_CSV_PATHS.items():
        try:
            subject_scores = extract_subject_scores(csv_path)
            for subject_id, score in subject_scores.items():
                plot_data.append({
                    'Model': model_name,
                    'Subject': subject_id,
                    'Correlation': score
                })
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        raise ValueError("No data could be loaded for plotting")
    
    # Use consistent styling with bar plot
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'grid.alpha': 0.3
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to only include models that have data
    preferred_order = ['BaselineDecoder2', 'Vlaai', 'Happyquoka2', 'TwoBranch_Recursive_3.0sec', 'TwoBranch_Recursive_3.0sec_GT']
    available_models = df['Model'].unique()
    model_order = [model for model in preferred_order if model in available_models]
    
    display_names = {
        'BaselineDecoder2': 'Linear', 'Vlaai': 'VLAAI', 'Happyquoka2': 'HappyQuoka',
        'TwoBranch_Recursive_3.0sec': 'DECAF',
        'TwoBranch_Recursive_3.0sec_GT': 'DECAF_Oracle'
    }
    
    # Plot individual subject lines (subset for clarity)
    subjects_to_plot = sorted(df['Subject'].unique())[:25]  # Plot first 25 subjects for better visibility
    
    for subject in subjects_to_plot:
        subject_data = df[df['Subject'] == subject]
        subject_data = subject_data.set_index('Model').reindex(model_order)
        
        x_pos = range(len(model_order))
        y_vals = [subject_data.loc[model, 'Correlation'] if model in subject_data.index else np.nan 
                  for model in model_order]
        
        ax.plot(x_pos, y_vals, 'o-', alpha=0.25, linewidth=0.8, markersize=3, color='gray')
    
    # Add mean lines with error bars
    means = []
    sems = []
    for i, model in enumerate(model_order):
        model_data = df[df['Model'] == model]['Correlation']
        mean_val = model_data.mean()
        sem_val = stats.sem(model_data.values)
        means.append(mean_val)
        sems.append(sem_val)
        
        ax.plot(i, mean_val, 'o', color=model_colors.get(model, '#333333'), 
                markersize=10, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    
    # Add error bars for means
    ax.errorbar(range(len(model_order)), means, yerr=sems, 
                fmt='none', color='black', capsize=4, capthick=1.5, zorder=4)
    
    # Styling consistent with bar plot
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([display_names[m] for m in model_order], rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Correlation Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    
    # Add subtle grid
    ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    y_min = df['Correlation'].min() - 0.02
    y_max = df['Correlation'].max() + 0.03
    ax.set_ylim(y_min, y_max)
    
    # Add group separator
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Add group labels
    ax.text(1, y_max - 0.01, 'Baseline Models', ha='center', va='top', 
            fontsize=10, style='italic', color='gray')
    ax.text(3.5, y_max - 0.01, 'Proposed Models', ha='center', va='top',
            fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='png')
        print(f"Subject trajectories plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_box_comparison(model_colors: Optional[Dict[str, str]] = None,
                       figsize: tuple = (10, 6),
                       save_path: Optional[str] = None,
                       show_plot: bool = True,
                       dpi: int = 300) -> plt.Figure:
    """
    Create box plots comparing model distributions.
    """
    
    if model_colors is None:
        model_colors = DEFAULT_MODEL_COLORS.copy()
    
    # Load all data
    plot_data = []
    for model_name, csv_path in MODEL_CSV_PATHS.items():
        try:
            subject_scores = extract_subject_scores(csv_path)
            for subject_id, score in subject_scores.items():
                plot_data.append({
                    'Model': model_name,
                    'Subject': subject_id,
                    'Correlation': score
                })
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        raise ValueError("No data could be loaded for plotting")
    
    # Use consistent styling with bar plot
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'grid.alpha': 0.3
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to only include models that have data
    preferred_order = ['BaselineDecoder2', 'Vlaai', 'Happyquoka2', 'TwoBranch_Recursive_3.0sec', 'TwoBranch_Recursive_3.0sec_GT']
    available_models = df['Model'].unique()
    model_order = [model for model in preferred_order if model in available_models]
    
    display_names = {
        'BaselineDecoder2': 'Linear', 'Vlaai': 'VLAAI', 'Happyquoka2': 'HappyQuoka',
        'TwoBranch_Recursive_3.0sec': 'DECAF',
        'TwoBranch_Recursive_3.0sec_GT': 'DECAF_Oracle'
    }
    
    # Create box plot data
    box_data = [df[df['Model'] == model]['Correlation'].values for model in model_order]
    colors = [model_colors.get(model, '#333333') for model in model_order]
    
    # Create box plot with consistent styling
    bp = ax.boxplot(box_data, patch_artist=True, 
                    showmeans=True, meanline=False, meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 6},
                    medianprops={'color': 'black', 'linewidth': 1.8},
                    whiskerprops={'color': 'black', 'linewidth': 1.2},
                    capprops={'color': 'black', 'linewidth': 1.2},
                    flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 4, 'alpha': 0.6})
    
    # Color the boxes to match other plots
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Styling consistent with bar plot
    ax.set_xticks(range(1, len(model_order) + 1))
    ax.set_xticklabels([display_names[m] for m in model_order], rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Correlation Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    
    # Add subtle grid
    ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set y-axis limits (add extra space at bottom for mean lines)
    y_min = df['Correlation'].min() - 0.08
    y_max = df['Correlation'].max() + 0.03
    ax.set_ylim(y_min, y_max)
    
    # Add mean value lines at the bottom for each model
    mean_line_y = y_min + 0.03  # Position for the mean lines
    for i, model in enumerate(model_order):
        model_data = df[df['Model'] == model]['Correlation']
        mean_val = model_data.mean()
        
        # Draw horizontal line showing the mean (box plot uses 1-based indexing)
        line_width = 0.4  # Width of the mean line
        ax.plot([i + 1 - line_width/2, i + 1 + line_width/2], [mean_line_y, mean_line_y], 
                color='black', linewidth=2.5, solid_capstyle='butt', zorder=10)
        
        # Add mean value text below the line
        ax.text(i + 1, mean_line_y - 0.015, f'{mean_val:.3f}', 
                ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Add group separator
    ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Add group labels
    ax.text(2, y_max - 0.01, 'Baseline Models', ha='center', va='top', 
            fontsize=10, style='italic', color='gray')
    ax.text(4.5, y_max - 0.01, 'Proposed Models', ha='center', va='top',
            fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='png')
        print(f"Box plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_difference_bars(model_colors: Optional[Dict[str, str]] = None,
                        figsize: tuple = (10, 6),
                        save_path: Optional[str] = None,
                        show_plot: bool = True,
                        dpi: int = 300) -> plt.Figure:
    """
    Create bar plot showing mean differences with confidence intervals.
    """
    
    if model_colors is None:
        model_colors = DEFAULT_MODEL_COLORS.copy()
    
    # Load all data
    all_scores = {}
    for model_name, csv_path in MODEL_CSV_PATHS.items():
        try:
            all_scores[model_name] = extract_subject_scores(csv_path)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    # Calculate means and confidence intervals
    # Only include models that have data loaded
    available_models = list(all_scores.keys())
    
    # Define the preferred order for models that exist
    preferred_order = ['BaselineDecoder2', 'Vlaai', 'Happyquoka2', 'TwoBranch_Recursive_3.0sec',
                      'TwoBranch_Recursive_3.0sec_GT']
    
    # Filter to only include models that actually have data
    model_order = [model for model in preferred_order if model in available_models]
    
    display_names = {
        'BaselineDecoder2': 'Linear', 'Vlaai': 'VLAAI', 'Happyquoka2': 'HappyQuoka',
        'TwoBranch_Recursive_3.0sec': 'DECAF',
        'TwoBranch_Recursive_3.0sec_GT': 'DECAF_Oracle'
    }
    
    means = []
    ci_lower = []
    ci_upper = []
    colors = []
    
    for model in model_order:
            scores = list(all_scores[model].values())
            mean_score = np.mean(scores)
            sem = stats.sem(scores)  # Standard error of mean
            ci = sem * stats.t.ppf(0.975, len(scores)-1)  # 95% CI
            
            means.append(mean_score)
            ci_lower.append(mean_score - ci)
            ci_upper.append(mean_score + ci)
            colors.append(model_colors.get(model, '#333333'))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(model_order))
    bars = ax.bar(x_pos, means, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.8)
    
    # Add error bars
    ax.errorbar(x_pos, means, 
                yerr=[np.array(means) - np.array(ci_lower), 
                      np.array(ci_upper) - np.array(means)],
                fmt='none', color='black', capsize=4, capthick=1.5)
    
    # Add value labels on bars (positioned at fixed height)
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                f'{mean_val:.3f}', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels([display_names[m] for m in model_order], rotation=45, ha='right')
    ax.set_ylabel('Mean Correlation Score', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add group separators
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Bar plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage - process all models and create violin plot
    print("Processing all models...")
    print("=" * 50)
    
    for model_name, csv_path in MODEL_CSV_PATHS.items():
        try:
            subject_scores = extract_subject_scores(csv_path)
            overall_mean = np.mean(list(subject_scores.values()))
            
            print(f"\n{model_name}:")
            print(f"  Overall mean score: {overall_mean:.4f}")
            print(f"  Total subjects: {len(subject_scores)}")
            
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("All models processed!")
    
    # Create bar plot
    print("\nCreating bar plot...")
    try:
        # Using default colors: baseline models in gray, recursive in green, context in blue
        fig = plot_difference_bars(
            model_colors=None,  # Use default colors (baselines: gray, recursive: green, context: blue)
            figsize=(10, 6),    # Good size for 5 models
            save_path='model_comparison_bar_plot.png',
            show_plot=False,    # Set to True to display the plot
            dpi=300            # High resolution for publication
        )
        
        print("✓ Bar plot created successfully!")

        # Create subject trajectories plot
        print("\nCreating subject trajectories plot...")
        fig = plot_subject_trajectories(
            model_colors=None,
            figsize=(10, 6),
            save_path='model_comparison_trajectories.png',
            show_plot=False,
            dpi=300
        )
        print("✓ Subject trajectories plot created successfully!")

        # Create box plot
        print("\nCreating box plot...")
        fig = plot_box_comparison(
            model_colors=None,
            figsize=(10, 6),
            save_path='model_comparison_box_plot.png',
            show_plot=False,
            dpi=300
        )
        print("✓ Box plot created successfully!")

        # Create violin plot
        print("\nCreating violin plot...")
        fig = plot_model_violin_comparison(
            model_colors=None,
            figsize=(12, 6),
            save_path='model_comparison_violin_plot.png',
            show_plot=False,
            dpi=300
        )
        print("✓ Violin plot created successfully!")
        
        
    except Exception as e:
        print(f"✗ Error creating bar plot: {e}")
    
    # Perform statistical comparison
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    results = compare_happyquoka_vs_eeg_pred()
    
    if results:
        print(f"\n📊 Summary for Paper:")
        print(f"HappyQuoka vs EEG-Pred comparison (n={results['n_subjects']}):")
        print(f"Mean difference: {results['difference']:.4f}")
        print(f"P-value: {results['p_value']:.6f}")
        print(f"Effect size: {results['effect_size']:.4f}")
        print(f"Result: {results['significance']}")