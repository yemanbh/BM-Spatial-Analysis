import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
import multiprocessing as mp
import numpy as np


def mark_cell_center(im, df, file_name, cell_2_color_dict=None, color='#00ff00'):
    
    # to be removed
    df = df.rename(columns={'class': 'Class'})
    dpi = 100
    height, width, nbands = im.shape

    # What size does the figure need to be in inches to fit the image?
    fig_size = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im, interpolation='nearest')

    # display cells on the image
    if cell_2_color_dict is not None:
        # if there are images without cells input df can be without column, 'Class'
        # this line adds Class column
        # if 'Class' not in list(df.columns):
        #     df['Class'] = 'CellA'
        sns.scatterplot(x='X',
                        y='Y',
                        data=df,
                        hue='Class',
                        ax=ax,
                        s=2,
                        edgecolor=None,
                        palette=cell_2_color_dict,
                        legend=False)
    else:

        # if 'Class' not in list(df.columns):
        #     df['Class'] = 'CellA'
        sns.scatterplot(x='X',
                        y='Y',
                        data=df,
                        ax=ax,
                        edgecolor=None,
                        palette=[color],
                        legend=False,
                        s=1)

    # Ensure we're displaying with square pixels and the right extent.
    # This is optional if you haven't called `plot` or anything else that might
    # change the limits/aspect.  We don't need this step in this case.
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    # save image with cell annotation
    fig.savefig(file_name, dpi=dpi, transparent=True)
    # plt.close(fig)
    plt.figure().clear()
    plt.clf()
    plt.close()


def map_cell_2_colour(cell_label_text):
    color_to_cell_dict = dict()
    
    with open(cell_label_text, 'r') as myfile:
        for line in myfile:
            color_val = line.split(' ')[0]
            color_val = color_val if color_val.startswith('#') else ''.join(['#', color_val])
            
            # the second split if the there is a comment without space
            cell_name = line.split(' ')[1].split('#')[0].strip()
            
            # change color to rgb
            # color_to_cell_dict[cell_name] = tuple(int(color_val[i:i + 2], 16) for i in (0, 2, 4))
            color_to_cell_dict[cell_name] = color_val
    
    return color_to_cell_dict


def annotate_images(output_dir,
                    images_dir,
                    csv_dir,
                    batch_name,
                    color_to_cell_dict,
                    file_names_list,
                    process_num,
                    image_ext):
    
    fig = plt.figure(num=1)
    
    for file_name in file_names_list:
        print('{};{};{}'.format(batch_name, file_name, process_num))
        f_name = os.path.splitext(file_name)[0]
        save_dir = os.path.join(output_dir, batch_name)
        os.makedirs(save_dir, exist_ok=True)
        output_file = os.path.join(save_dir, f_name + '.jpg')
        
        # check if file exists
        if os.path.isfile(output_file):
            print('{} already exists'.format(output_file))
            continue

        df = pd.read_csv(os.path.join(csv_dir, batch_name, file_name))
        df = df.rename(columns={'class': 'Class'})

        image_found = False
        for ext in image_ext:
            image_path = os.path.join(images_dir, batch_name, f_name + ext)
            if os.path.isfile(image_path):
                image_found = True
                break
        assert image_found is True

        im = io.imread(image_path)

        mark_cell_center_new(im=im,
                             df=df,
                             fig=fig,
                             cell_2_color_dict=color_to_cell_dict,
                             file_name=output_file)
        del im
        del df

    plt.close(fig)


def mark_cell_center_new(im, df, file_name, fig, cell_2_color_dict=None):
    dpi = 100
    height, width = im.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    fig_size = width / float(dpi), height / float(dpi)
    
    # Create a figure of the right size with one axes that takes up the full figure
    # fig = plt.figure(figsize=fig_size)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Hide spines, ticks, etc.
    ax.axis('off')
    
    # Display the image.
    ax.imshow(im, interpolation='nearest')
    
    # display cells on the image
    if cell_2_color_dict is not None:
        # if there are images without cells input df can be without column, 'Class'
        # this line adds Class column
        if 'Class' not in list(df.columns):
            df['Class'] = 'CellA'
        sns.scatterplot(x='X',
                        y='Y',
                        data=df,
                        hue='Class',
                        ax=ax,
                        s=8,
                        edgecolor=None,
                        palette=cell_2_color_dict,
                        legend=False)
    else:
        
        # if 'Class' not in list(df.columns):
        #     df['Class'] = 'CellA'
        sns.scatterplot(x='X',
                        y='Y',
                        data=df,
                        ax=ax,
                        edgecolor=None,
                        legend=False,
                        s=8)
    
    # Ensure we're displaying with square pixels and the right extent.
    # This is optional if you haven't called `plot` or anything else that might
    # change the limits/aspect.  We don't need this step in this case.
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    
    # save image with cell annotation
    fig.savefig(file_name, dpi=dpi, transparent=True)
    
    # plt.figure().clear()
    ax.remove()
    plt.cla()
    plt.clf()


def run_multi_process(output_dir,
                      images_dir,
                      csv_dir,
                      batch_name,
                      img_files_names_list,
                      num_processes,
                      image_ext,
                      color_to_cell_dict):

    n = len(img_files_names_list)

    if n < num_processes:
        num_processes = n

    num_elem_per_process = int(np.ceil(n / num_processes))

    file_names_list_list = []

    for i in range(num_processes):
        start_ = i * num_elem_per_process
        file_names_list_list.append(img_files_names_list[start_: start_ + num_elem_per_process])

    # create list of processes
    processes = [
        mp.Process(target=annotate_images, args=(output_dir,
                                                 images_dir,
                                                 csv_dir,
                                                 batch_name,
                                                 color_to_cell_dict,
                                                 file_names_list_list[process_num],
                                                 process_num,
                                                 image_ext))
        for
        process_num in range(num_processes)]

    print('{} processes created'.format(num_processes))

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()
    print('All Processes finished!!!')
    

def run(output_dir,
        images_dir,
        csv_dir,
        image_ext,
        slide_name,
        color_to_cell_dict=None,
        num_processes=1,

        img_name_pattern=None):

    if not os.path.isdir(os.path.join(csv_dir, slide_name)):
        return 0
    img_files_list_all = os.listdir(os.path.join(csv_dir, slide_name))

    if img_name_pattern is None:
        img_files_names_list = img_files_list_all
    else:
        img_files_names_list = [file_name for file_name in
                                img_files_list_all if any([x in file_name for x in img_name_pattern])
                                is True]
    if num_processes > 1:
        run_multi_process(output_dir=output_dir,
                          images_dir=images_dir,
                           csv_dir=csv_dir,
                           image_ext=image_ext,
                           batch_name=slide_name,
                          num_processes=num_processes,
                          color_to_cell_dict=color_to_cell_dict,
                           img_files_names_list=img_files_names_list)
    else:
        annotate_images(output_dir=output_dir,
                        images_dir=images_dir,
                        csv_dir=csv_dir,
                        color_to_cell_dict=color_to_cell_dict,
                        batch_name=slide_name,
                        file_names_list=img_files_names_list,
                        process_num=0,
                        image_ext=image_ext)
