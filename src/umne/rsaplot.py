"""
RSA plotting functions

Authors: Dror Dotan <dror.dotan@gmail.com>
         Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""
import matplotlib.animation as animation
from matplotlib import pyplot as plt


#--------------------------------------------------------------------
def init_ffmpeg_path(path=u'/usr/local/bin/ffmpeg'):
    """
    Initialize the absolute path to FFMPEG
    """
    plt.rcParams['animation.ffmpeg_path'] = path


#--------------------------------------------------------------------
def movie(data, vmin, vmax, save_name,times, x_ticks,labels):
    """
    does a movie

    :param data to put in the movie of shape n_subjects X n_positions X n_times
    :param save_name: the name of the saving file
    :return: True
    """
    # build the movies

    fig, (ax) = plt.subplots(1, 1, figsize=(8, 8))
    n_times = len(data)
    im = ax.imshow(data[0], interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    def update(dictio):
        ax.cla()  # <-- clear the subplot otherwise boxplot shows previous frame
        t = dictio['t']
        datat = dictio['data']
        string = " Time in epoch %i ms" % int(times[t])
        ax.set_title(string)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(x_ticks)
        ax.set_yticklabels(labels)
        ax.imshow(datat,interpolation='none', vmin=vmin, vmax=vmax)

    def data_gen():
        t = 0
        while t<n_times:
            yield {'data':data[t],'t':t}
            t += 1

    # noinspection PyTypeChecker
    ani = animation.FuncAnimation(fig, update, data_gen, interval=50)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

    ani.save(save_name + '.mp4', writer=writer)

    return ani

