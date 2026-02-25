import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 8))  # REDUCED from (14, 10)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'
color_lstm = '#FFF4E6'
color_dense = '#F0E6FF'
color_output = '#E8F8E8'

# Helper function to draw boxes
def draw_box(ax, x, y, width, height, text, color, fontsize=10, fontweight='normal'):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize, fontweight=fontweight)

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=1.5, color='black')
    ax.add_patch(arrow)

# ============================================
# INPUT LAYERS
# ============================================

# Sequence input - showing 5 timesteps with MORE SPACING
seq_x = 0.5
seq_y_base = 8.5
seq_width = 1.2
seq_height = 0.7
seq_spacing = 1

timesteps = ['t-20', 't-15', 't-10', 't-5', 't']
for i, ts in enumerate(timesteps):
    y_pos = seq_y_base - i * seq_spacing
    draw_box(ax, seq_x, y_pos, seq_width, seq_height,
             f'{ts}\n6 features\n(PC₁₋₄, RMSE, IV)',
             color_input, fontsize=10)  # INCREASED from 8
    
# Label for sequence input
ax.text(seq_x + seq_width/2, seq_y_base + 0.9, 'Sequence Input',
        ha='center', fontsize=13, fontweight='bold')  # INCREASED from 11

# Static input
static_x = 0.5
static_y = 0.5
draw_box(ax, static_x, static_y -0.1, seq_width, seq_height,
         'Eigenvalues\nλ₁, λ₂, λ₃, λ₄',
         color_input, fontsize=10)  # INCREASED from 8
ax.text(static_x + seq_width/2, static_y + 0.8, 'Static Input',
        ha='center', fontsize=13, fontweight='bold')  # INCREASED from 11

# ============================================
# LSTM LAYER
# ============================================

lstm_x = 3.0
lstm_y = 6.0
lstm_width = 1.5
lstm_height = 2.5

draw_box(ax, lstm_x, lstm_y, lstm_width, lstm_height,
         'LSTM\n20 units\n\nActivation: tanh\nRecurrent: sigmoid',
         color_lstm, fontsize=11, fontweight='bold')  # INCREASED from 9

# Arrows from sequence to LSTM
for i in range(5):
    y_pos = seq_y_base - i * seq_spacing + seq_height/2
    draw_arrow(ax, seq_x + seq_width, y_pos, lstm_x - 0.1, lstm_y + lstm_height/2)

# ============================================
# CONCATENATE
# ============================================

concat_x = 5.5
concat_y = 6.5
concat_width = 1.2
concat_height = 1.5

draw_box(ax, concat_x, concat_y, concat_width - 0.1, concat_height,
         'Concatenate\n\n20 + 4 = 24',
         '#FFE6E6', fontsize=11, fontweight='bold')  # INCREASED from 9

# Arrow from LSTM to Concatenate
draw_arrow(ax, lstm_x + lstm_width, lstm_y + lstm_height/2,
           concat_x - 0.1, concat_y + concat_height/2)

# Arrow from Static to Concatenate
draw_arrow(ax, static_x + seq_width, static_y + seq_height/2,
           concat_x - 0.1, concat_y + concat_height/2)

# ============================================
# DENSE LAYER
# ============================================

dense_x = 7.5
dense_y = 6.5
dense_width = 1.2
dense_height = 1.5

draw_box(ax, dense_x, dense_y, dense_width, dense_height,
         'Dense\n16 units\n\nReLU',
         color_dense, fontsize=11, fontweight='bold')  # INCREASED from 9

draw_arrow(ax, concat_x + concat_width, concat_y + concat_height/2,
           dense_x - 0.1, dense_y + dense_height/2)

# ============================================
# DROPOUT
# ============================================

dropout_x = 7.5
dropout_y = 4.5
dropout_width = 1.2
dropout_height = 0.8

draw_box(ax, dropout_x, dropout_y, dropout_width, dropout_height,
         'Dropout\nrate = 0.2',
         '#F5F5F5', fontsize=10)  # INCREASED from 8

draw_arrow(ax, dense_x + dense_width/2, dense_y,
           dropout_x + dropout_width/2, dropout_y + dropout_height + 0.1)

# ============================================
# OUTPUT LAYER
# ============================================

output_x = 7.5
output_y = 1.0
output_width = 1.2
output_height = 2

draw_box(ax, output_x, output_y, output_width, output_height,
         'Output\n1 unit\n\nSigmoid\n\nBinary\nPC direction: [1,0]',
         color_output, fontsize=11, fontweight='bold')  # INCREASED from 9

draw_arrow(ax, dropout_x + dropout_width/2, dropout_y,
           output_x + output_width/2, output_y + output_height + 0.1)

# ============================================
# SAVE
# ============================================

plt.tight_layout()
plt.savefig('lstm_architecture.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: lstm_architecture.png")