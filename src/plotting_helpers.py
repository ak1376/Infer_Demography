import colorsys
import matplotlib.pyplot as plt
import math
import numpy as np

def create_color_scheme(num_params):

    main_colors = []
    color_shades = {}
    
    for i in range(num_params):
        # Generate main color using HSV color space
        hue = i / num_params
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        
        # Convert RGB to hex
        main_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        main_colors.append(main_color)
        
        # Generate shades
        shades = []
        for j in range(3):
            # Adjust saturation and value for shades
            sat = 1.0 - (j * 0.3)
            val = 1.0 - (j * 0.2)
            shade_rgb = colorsys.hsv_to_rgb(hue, sat, val)
            shade = '#{:02x}{:02x}{:02x}'.format(int(shade_rgb[0]*255), int(shade_rgb[1]*255), int(shade_rgb[2]*255))
            shades.append(shade)
        
        color_shades[main_color] = shades

    return color_shades, main_colors


def visualizing_results(
    linear_mdl_obj, analysis, save_loc="results", outlier_indices=None, stages=None, color_shades=None, main_colors=None
):
    # Default to all stages if not specified
    if stages is None:
        stages = ["training", "validation", "testing"]

    params = linear_mdl_obj['param_names']
    num_params = linear_mdl_obj['training']['targets'].shape[1]

    rows = math.ceil(math.sqrt(num_params))
    cols = math.ceil(num_params / rows)

    plt.figure(figsize=(5 * cols, 5 * rows))

    # Loop through each parameter (dimension of the parameters)
    for i, param in enumerate(params):
        plt.subplot(rows, cols, i + 1)

        all_predictions = []
        all_targets = []

        # Loop through each stage and parameter combination
        for j, stage in enumerate(stages):
            # Ensure predictions are NumPy arrays
            if not isinstance(linear_mdl_obj[stage]["predictions"], np.ndarray):
                linear_mdl_obj[stage]["predictions"] = linear_mdl_obj[stage]["predictions"].to_numpy()

            # Ensure targets are NumPy arrays
            if not isinstance(linear_mdl_obj[stage]["targets"], np.ndarray):
                linear_mdl_obj[stage]["targets"] = linear_mdl_obj[stage]["targets"].to_numpy()

            predictions = linear_mdl_obj[stage]["predictions"][:, i]  # Only for the current parameter
            targets = linear_mdl_obj[stage]["targets"][:, i]  # Only for the current parameter

            # Append current parameter's predictions and targets to global lists for axis scaling
            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # Scatter plot for each stage
            plt.scatter(
                targets,
                predictions,
                alpha=0.5,
                color=color_shades[main_colors[i % len(main_colors)]][j],  # Color for each stage #type:ignore 
                label=f"{stage.capitalize()}",
            )

        # Set equal axis limits based on global min/max for each parameter's predictions and targets
        max_value = max(max(all_predictions), max(all_targets))
        min_value = min(min(all_predictions), min(all_targets))
        
        plt.xlim([min_value, max_value])
        plt.ylim([min_value, max_value])

        # Add the ideal line y = x
        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="black",
            linestyle="--",
            label="Ideal: Prediction = Target",
        )

        # Set equal aspect ratio
        plt.gca().set_aspect('equal', 'box')

        # Labels and title
        plt.xlabel(f"True {param}")
        plt.ylabel(f"Predicted {param}")
        plt.title(f"{param}: True vs Predicted")

        plt.legend()

    plt.tight_layout()

    # Save the figure
    filename = f"{save_loc}/{analysis}"
    if outlier_indices is not None:
        filename += "_outliers_removed"
    filename += ".png"

    plt.savefig(filename, format="png", dpi=300)
    plt.show()
