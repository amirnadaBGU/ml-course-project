import matplotlib

# Make window interactive
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Qt5Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_image_objects(image_name, df, original_coordinates=False):
    """
    function that receives an image name and a dataframe with object keypoints,
    and plots the objects on a matplotlib interactive window.
    """

    # data filtering
    img_df = df[df['image_stem'] == image_name]

    if img_df.empty:
        print(f"Image {image_name} not found.")
        return

    # resolution settings
    if original_coordinates:
        img_width = 640
        img_height = 360
    else:
        img_width = 1
        img_height = 1

    # plotting
    fig, ax = plt.subplots(figsize=(12, 6) if original_coordinates else (10, 10))
    ax.set_aspect('equal')

    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Y axis inverted for image coordinates

    objects = img_df['object_id'].unique()
    cmap = plt.get_cmap('tab20')

    for i, obj_id in enumerate(objects):
        obj_points = img_df[img_df['object_id'] == obj_id].sort_values('keypoint_index')

        x = obj_points['x_norm'].values * img_width
        y = obj_points['y_norm'].values * img_height

        color = cmap(i % 20)

        if len(x) == 4:
            # 3->0, 0->1, 1->2
            for a, b in [(3, 0), (0, 1), (1, 2)]:
                ax.plot([x[a], x[b]], [y[a], y[b]], color=color, linewidth=2, solid_capstyle='round')

            ax.scatter(x, y, color=color, s=30)

            for k in range(len(x)):
                ax.text(x[k], y[k], str(k), color=color, fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        else:
            ax.scatter(x, y, color=color, label=f'Obj {obj_id}')

    title_suffix = " (Original Coords)" if original_coordinates else " (Normalized)"
    plt.title(f'Visualization of {image_name}\n({len(objects)} objects){title_suffix}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.6)

    print("Opening interactive plot window...")
    plt.show(block=True)


# Main Procedure
if __name__ == "__main__":
    file_path = 'train_data.xlsx'

    # Data loading
    try:
        df = pd.read_excel(file_path)

        if not df.empty:
            # Choose the image with maximal number of points
            image_counts = df['image_stem'].value_counts()
            example_image = image_counts.idxmax()

            print(f"Plotting image: {example_image}")
            plot_image_objects(example_image, df, original_coordinates=True)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")