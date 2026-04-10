import json
import glob
import matplotlib.pyplot as plt
from collections import Counter
import os

def visualize_corrections():
    db_files = glob.glob("db_*.json")
    
    if not db_files:
        print("No db_*.json files found.")
        return

    transitions = []
    reviewer_corrections = []
    
    for file_path in db_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = data.get("items", [])
            for item in items:
                review = item.get("review", {})
                if review.get("status") == "reviewed" and review.get("decision") == "reject":
                    original_label = item.get("llm", {}).get("label", "Unknown")
                    final_label = review.get("final_label", "Unknown")
                    reviewer = review.get("reviewed_by", "Unknown")
                    
                    transition = f"{original_label} -> {final_label}"
                    transitions.append(transition)
                    reviewer_corrections.append(reviewer)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not transitions:
        print("No corrections (rejections) found to visualize.")
        return
        
    # Count transitions
    transition_counts = Counter(transitions)
    # Count reviewer corrections
    reviewer_counts = Counter(reviewer_corrections)
    
    sorted_transitions = transition_counts.most_common()
    sorted_reviewers = reviewer_counts.most_common()

    # Set up the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Transitions (Camp 1 -> Camp 2)
    t_labels = [t[0] for t in sorted_transitions]
    t_values = [t[1] for t in sorted_transitions]
    
    # Use horizontal bar chart so labels are readable
    bars1 = ax1.barh(t_labels, t_values, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Corrections')
    ax1.set_title('Corrections: Original -> Final Label (Camp 1 -> Camp 2)')
    ax1.invert_yaxis() # Put highest counts at the top
    
    # Add numerical labels on the bars
    for bar in bars1:
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                 f'{int(bar.get_width())}', 
                 va='center', ha='left')
    ax1.set_xlim(0, max(t_values) + max(t_values)*0.15)

    # Plot 2: Reviewer Rankings
    r_labels = [r[0].capitalize() for r in sorted_reviewers]
    r_values = [r[1] for r in sorted_reviewers]
    
    bars2 = ax2.bar(r_labels, r_values, color='lightcoral', edgecolor='black')
    ax2.set_ylabel('Number of Corrections')
    ax2.set_title('Ranked Corrections by Reviewer')
    
    # Add numerical labels on top of the bars
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                 f'{int(bar.get_height())}', 
                 va='bottom', ha='center')
    ax2.set_ylim(0, max(r_values) + max(r_values)*0.15)

    plt.tight_layout()
    output_img = "corrections_visualization.png"
    plt.savefig(output_img, dpi=300)
    print(f"Visualization successfully generated and saved to {output_img}")

if __name__ == "__main__":
    visualize_corrections()
