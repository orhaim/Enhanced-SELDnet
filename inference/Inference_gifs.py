import imageio
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from keras_self_attention import SeqSelfAttention
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from matplotlib import cm
import pandas as pd


def _split_in_seqs(data):
    _seq_len = 64
    if len(data.shape) == 1:
        if data.shape[0] % _seq_len:
            data = data[:-(data.shape[0] % _seq_len), :]
        data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % _seq_len:
            data = data[:-(data.shape[0] % _seq_len), :]
        data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % _seq_len:
            data = data[:-(data.shape[0] % _seq_len), :, :]
        data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
    else:
        print('ERROR: Unknown data dimensions: {}'.format(data.shape))
        exit()
    return data


def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


my_model = load_model('C:/Users/Meital/PycharmProjects/Drones/attention_3_ansim_ov1_split2_regr0_3d0_1_model.h5', custom_objects=SeqSelfAttention.get_custom_objects())
my_model.summary()
feat = np.load('C:/Users/Meital/PycharmProjects/Drones/test/ov1/train_6_desc_30_100.wav.npy')
feat = np.reshape(feat, (5166, 256, 8))
feat = _split_in_seqs(feat)
feat = np.transpose(feat, (0, 3, 1, 2))
pred = my_model.predict(feat)
sed = (pred[0].reshape(pred[0].shape[0] * pred[0].shape[1], pred[0].shape[2]) > 0.5).astype(int)
doa = pred[1].reshape(pred[1].shape[0] * pred[1].shape[1], pred[1].shape[2])

label = np.load('C:/Users/Meital/PycharmProjects/Drones/test/ov1/label_train_6_desc_30_100.wav.npy')
label = _split_in_seqs(label)
label_sed = label[:, :, :11]
label_doa = label[:, :, 11:]
label_sed = (label_sed.reshape(label_sed.shape[0] * label_sed.shape[1], label_sed.shape[2])).astype(int)
label_doa = label_doa.reshape(label_doa.shape[0] * label_doa.shape[1], label_doa.shape[2]) * np.pi / 180

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.75, 0.75)
ax.set_ylim(-0.75, 0.75)
ax.set_zlim(-0.75, 0)
ax.set_title('3D Plot - 1 Overlap')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# fig_labels = plt.figure()
# ax_labels = fig_labels.add_subplot(111, projection='3d')
# ax_labels.set_xlim(-0.75, 0.75)
# ax_labels.set_ylim(-0.75, 0.75)
# ax_labels.set_zlim(-0.75, 0.75)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.set_xlim(-0.75, 0.75)
# ax1.set_ylim(-0.75, 0.75)
# ax1.set_zlim(-0.75, 0.75)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.set_xlim(-0.75, 0.75)
# ax2.set_ylim(-0.75, 0.75)
# ax2.set_zlim(-0.75, 0.75)
# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111, projection='3d')
# ax3.set_xlim(-0.75, 0.75)
# ax3.set_ylim(-0.75, 0.75)
# ax3.set_zlim(-0.75, 0.75)
# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111, projection='3d')
# ax4.set_xlim(-0.75, 0.75)
# ax4.set_ylim(-0.75, 0.75)
# ax4.set_zlim(-0.75, 0.75)
# fig5 = plt.figure()
# ax5 = fig5.add_subplot(111, projection='3d')
# ax5.set_xlim(-0.75, 0.75)
# ax5.set_ylim(-0.75, 0.75)
# ax5.set_zlim(-0.75, 0.75)
# fig6 = plt.figure()
# ax6 = fig6.add_subplot(111, projection='3d')
# ax6.set_xlim(-0.75, 0.75)
# ax6.set_ylim(-0.75, 0.75)
# ax6.set_zlim(-0.75, 0.75)
# fig7 = plt.figure()
# ax7 = fig7.add_subplot(111, projection='3d')
# ax7.set_xlim(-0.75, 0.75)
# ax7.set_ylim(-0.75, 0.75)
# ax7.set_zlim(-0.75, 0.75)
# fig8 = plt.figure()
# ax8 = fig8.add_subplot(111, projection='3d')
# ax8.set_xlim(-0.75, 0.75)
# ax8.set_ylim(-0.75, 0.75)
# ax8.set_zlim(-0.75, 0.75)
# fig9 = plt.figure()
# ax9 = fig9.add_subplot(111, projection='3d')
# ax9.set_xlim(-0.75, 0.75)
# ax9.set_ylim(-0.75, 0.75)
# ax9.set_zlim(-0.75, 0.75)
# fig10 = plt.figure()
# ax10 = fig10.add_subplot(111, projection='3d')
# ax10.set_xlim(-0.75, 0.75)
# ax10.set_ylim(-0.75, 0.75)
# ax10.set_zlim(-0.75, 0.75)
# fig11 = plt.figure()
# ax11 = fig11.add_subplot(111, projection='3d')
# ax11.set_xlim(-0.75, 0.75)
# ax11.set_ylim(-0.75, 0.75)
# ax11.set_zlim(-0.75, 0.75)
# fig12 = plt.figure()
# ax12 = fig12.add_subplot(111, projection='3d')
# ax12.set_xlim(-0.75, 0.75)
# ax12.set_ylim(-0.75, 0.75)
# ax12.set_zlim(-0.75, 0.75)
# fig13 = plt.figure()
# ax13 = fig13.add_subplot(111, projection='3d')
# ax13.set_xlim(-0.75, 0.75)
# ax13.set_ylim(-0.75, 0.75)
# ax13.set_zlim(-0.75, 0.75)
# fig14 = plt.figure()
# ax14 = fig14.add_subplot(111, projection='3d')
# ax14.set_xlim(-0.75, 0.75)
# ax14.set_ylim(-0.75, 0.75)
# ax14.set_zlim(-0.75, 0.75)
# fig15 = plt.figure()
# ax15 = fig15.add_subplot(111, projection='3d')
# ax15.set_xlim(-0.75, 0.75)
# ax15.set_ylim(-0.75, 0.75)
# ax15.set_zlim(-0.75, 0.75)
# fig16 = plt.figure()
# ax16 = fig16.add_subplot(111, projection='3d')
# ax16.set_xlim(-0.75, 0.75)
# ax16.set_ylim(-0.75, 0.75)
# ax16.set_zlim(-0.75, 0.75)
# fig17 = plt.figure()
# ax17 = fig17.add_subplot(111, projection='3d')
# ax17.set_xlim(-0.75, 0.75)
# ax17.set_ylim(-0.75, 0.75)
# ax17.set_zlim(-0.75, 0.75)
# fig18 = plt.figure()
# ax18 = fig18.add_subplot(111, projection='3d')
# ax18.set_xlim(-0.75, 0.75)
# ax18.set_ylim(-0.75, 0.75)
# ax18.set_zlim(-0.75, 0.75)
# fig19 = plt.figure()
# ax19 = fig19.add_subplot(111, projection='3d')
# ax19.set_xlim(-0.75, 0.75)
# ax19.set_ylim(-0.75, 0.75)
# ax19.set_zlim(-0.75, 0.75)
# fig20 = plt.figure()
# ax20 = fig20.add_subplot(111, projection='3d')
# ax20.set_xlim(-0.75, 0.75)
# ax20.set_ylim(-0.75, 0.75)
# ax20.set_zlim(-0.75, 0.75)
# fig21 = plt.figure()
# ax21 = fig21.add_subplot(111, projection='3d')
# ax21.set_xlim(-0.75, 0.75)
# ax21.set_ylim(-0.75, 0.75)
# ax21.set_zlim(-0.75, 0.75)
# fig22 = plt.figure()
# ax22 = fig22.add_subplot(111, projection='3d')
# ax22.set_xlim(-0.75, 0.75)
# ax22.set_ylim(-0.75, 0.75)
# ax22.set_zlim(-0.75, 0.75)
# fig23 = plt.figure()
# ax23 = fig23.add_subplot(111, projection='3d')
# ax23.set_xlim(-0.75, 0.75)
# ax23.set_ylim(-0.75, 0.75)
# ax23.set_zlim(-0.75, 0.75)
# fig24 = plt.figure()
# ax24 = fig24.add_subplot(111, projection='3d')
# ax24.set_xlim(-0.75, 0.75)
# ax24.set_ylim(-0.75, 0.75)
# ax24.set_zlim(-0.75, 0.75)

colors = {0: ['c', "."], 1: ['b', "v"], 2: ['g', "^"], 3: ['yellow', "<"], 4: ['m', ">"], 5: ['purple', "s"], 6: ['r', "P"], 7: ['violet', "D"], 8: ['peru', "*"], 9: ['silver', "X"], 10: ['palegreen', "d"]}
# for i, col in enumerate(np.transpose(sed)):
#     for j, row in enumerate(col):
#         if row == 1:
#             x, y, z = doa[j, i], doa[j, i + 11], doa[j, i + 22]
#             ax.scatter(x, y, z, c=colors[i][0], s=2)
# for i, col in enumerate(np.transpose(label_sed)):
#     for j, row in enumerate(col):
#         if row == 1:
#             x, y, z = sph2cart(label_doa[j, i], label_doa[j, i+11], 1)
#             ax_labels.scatter(x, y, z, c=colors[i][0], s=2)
#
# plt.show()

class_num = 10
const = 1000
low, high = 3000, 4000
for j, col in enumerate(np.transpose(sed)):
    for i, row in enumerate(col[low:high]):
        if row == 1:
            # if j == class_num:
                x, y, z = doa[i, j], doa[i, j+11], doa[i, j+22]
                ax.scatter(x, y, z, c=colors[j][0], s=3)
                x, y, z = sph2cart(label_doa[i, j], label_doa[i, j + 11], 1)
                if -0.55 <= x <= -0.45:
                    x, y, _ = sph2cart(label_doa[i, j] + (5 * 180 / np.pi), label_doa[i, j + 11] + (5 * 180 / np.pi), 1)
                    z -= 0.86
                ax.scatter(x, y, z, c=colors[j][0], s=400, alpha=0.1, marker='+')
                # if i < const:
                #     ax.scatter(x, y, z, c=colors[j][0], s=1)#, marker=colors[j][1])
                # elif const <= i < const*2:
                #     ax1.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*2 <= i < const*3:
                #     ax2.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*3 <= i < const*4:
                #     ax3.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*4 <= i < const*5:
                #     ax4.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*5 <= i < const*6:
                #     ax5.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*6 <= i < const*7:
                #     ax6.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*7 <= i < const*8:
                #     ax7.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*8 <= i < const*9:
                #     ax8.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*9 <= i < const*10:
                #     ax9.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*10 <= i < const*11:
                #     ax10.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*11 <= i < const*12:
                #     ax11.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*12 <= i < const*13:
                #     ax12.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*13 <= i < const*14:
                #     ax13.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*14 <= i < const*15:
                #     ax14.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*15 <= i < const*16:
                #     ax15.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*16 <= i < const*17:
                #     ax16.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*17 <= i < const*18:
                #     ax17.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*18 <= i < const*19:
                #     ax18.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*19 <= i < const*20:
                #     ax19.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*20 <= i < const*21:
                #     ax20.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*21 <= i < const*22:
                #     ax21.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*22 <= i < const*23:
                #     ax22.scatter(x, y, z, c=colors[j][0], s=1)
                # elif const*23 <= i < const*24:
                #     ax23.scatter(x, y, z, c=colors[j][0], s=1)
                # else:
                #     ax24.scatter(x, y, z, c=colors[j][0], s=1)
for angle in range(0, 360, 15):
    ax.view_init(60, angle)
    plt.draw()
    plt.savefig(str(angle) + '.png')
    plt.pause(.0001)
# for j, col in enumerate(np.transpose(label_sed)):
#     for i, row in enumerate(col[low:high]):
#         if row == 1:
#             # if j == class_num:
#                 x, y, z = sph2cart(label_doa[i, j], label_doa[i, j+11], 1)
#                 if -0.55 <= x <= -0.45:
#                     x, y, _ = sph2cart(label_doa[i, j] + (5 * 180 / np.pi), label_doa[i, j + 11] + (5 * 180 / np.pi), 1)
#                     z -= 0.86
                # if i < const:
                #     ax.scatter(x, y, z, c='k', s=2)  # , marker=colors[j][1])
                # elif const <= i < const * 2:
                #     ax1.scatter(x, y, z, c='k', s=2)
                # elif const * 2 <= i < const * 3:
                #     ax2.scatter(x, y, z, c='k', s=2)
                # elif const * 3 <= i < const * 4:
                #     ax3.scatter(x, y, z, c='k', s=2)
                # elif const * 4 <= i < const * 5:
                #     ax4.scatter(x, y, z, c='k', s=2)
                # elif const * 5 <= i < const * 6:
                #     ax5.scatter(x, y, z, c='k', s=2)
                # elif const * 6 <= i < const * 7:
                #     ax6.scatter(x, y, z, c='k', s=2)
                # elif const * 7 <= i < const * 8:
                #     ax7.scatter(x, y, z, c='k', s=2)
                # elif const * 8 <= i < const * 9:
                #     ax8.scatter(x, y, z, c='k', s=2)
                # elif const * 9 <= i < const * 10:
                #     ax9.scatter(x, y, z, c='k', s=2)
                # elif const * 10 <= i < const * 11:
                #     ax10.scatter(x, y, z, c='k', s=2)
                # elif const * 11 <= i < const * 12:
                #     ax11.scatter(x, y, z, c='k', s=2)
                # elif const * 12 <= i < const * 13:
                #     ax12.scatter(x, y, z, c='k', s=2)
                # elif const * 13 <= i < const * 14:
                #     ax13.scatter(x, y, z, c='k', s=2)
                # elif const * 14 <= i < const * 15:
                #     ax14.scatter(x, y, z, c='k', s=2)
                # elif const * 15 <= i < const * 16:
                #     ax15.scatter(x, y, z, c='k', s=2)
                # elif const * 16 <= i < const * 17:
                #     ax16.scatter(x, y, z, c='k', s=2)
                # elif const * 17 <= i < const * 18:
                #     ax17.scatter(x, y, z, c='k', s=2)
                # elif const * 18 <= i < const * 19:
                #     ax18.scatter(x, y, z, c='k', s=2)
                # elif const * 19 <= i < const * 20:
                #     ax19.scatter(x, y, z, c='k', s=2)
                # elif const * 20 <= i < const * 21:
                #     ax20.scatter(x, y, z, c='k', s=2)
                # elif const * 21 <= i < const * 22:
                #     ax21.scatter(x, y, z, c='k', s=2)
                # elif const * 22 <= i < const * 23:
                #     ax22.scatter(x, y, z, c='k', s=2)
                # elif const * 23 <= i < const * 24:
                #     ax23.scatter(x, y, z, c='k', s=2)
                # else:
                #     ax24.scatter(x, y, z, c='k', s=2)

            # if j == 0:
            #     ax1.scatter(x, y, z, c='b', s=2)
            # if j == 1:
            #     ax2.scatter(x, y, z, c='b', s=2)
            # if j == 2:
            #     ax3.scatter(x, y, z, c='b', s=2)
            # if j == 3:
            #     ax4.scatter(x, y, z, c='b', s=2)
            # if j == 4:
            #     ax5.scatter(x, y, z, c='b', s=2)
            # if j == 5:
            #     ax6.scatter(x, y, z, c='b', s=2)
            # if j == 6:
            #     ax7.scatter(x, y, z, c='b', s=2)
            # if j == 7:
            #     ax8.scatter(x, y, z, c='b', s=2)
            # if j == 8:
            #     ax9.scatter(x, y, z, c='b', s=2)
            # if j == 9:
            #     ax10.scatter(x, y, z, c='b', s=2)
            # if j == 10:
            #     ax11.scatter(x, y, z, c='b', s=2)

# plots_list = [fig, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19, fig20, fig21, fig22, fig23, fig24]
# for i, plot in enumerate(plots_list):
#     plot.savefig('class_' + str(class_num) + '_fig_' + str(i) + '.png')
# #
# for i, row in enumerate(label_sed):
#     for j, col in enumerate(row):
#         if col == 1:
#             x, y, z = sph2cart(label_doa[i, j], label_doa[i, j+11], 1)
#             ax_labels.scatter(x, y, z, c=colors[j][0], s=2)    # , marker=colors[j][1])
#
#             if j == 0:
#                 ax1.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 1:
#                 ax2.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 2:
#                 ax3.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 3:
#                 ax4.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 4:
#                 ax5.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 5:
#                 ax6.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 6:
#                 ax7.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 7:
#                 ax8.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 8:
#                 ax9.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 9:
#                 ax10.scatter(x, y, z, c='y', s=2, alpha=0.8)
#             if j == 10:
#                 ax11.scatter(x, y, z, c='y', s=2, alpha=0.8)

# plt.show()
# x_data, y_data, z_data = [], [], []
# for i, row in enumerate(sed):
#     for j, col in enumerate(row):
#         if col == 1:
#             x, y, z = doa[i, j], doa[i, j+11], doa[i, j+22]
#             if j == 2:
#                 x_data.append(x)
#                 y_data.append(y)
#                 z_data.append(z)
            # ax.scatter(x, y, z, c=colors[j][0], s=2)    # , marker=colors[j][1])
            # colors[j][2].scatter(x, y, z, c='b', s=2)

# xModel = np.linspace(min(x_data), max(x_data), 20)
# yModel = np.linspace(min(y_data), max(y_data), 20)
# X, Y = np.meshgrid(xModel, yModel)
#
#
# def func(data, a, alpha, beta):
#     x = data[0]
#     y = data[1]
#     return a * (x**alpha) * (y**beta)
#
#
# fittedParameters, pcov = scipy.optimize.curve_fit(func, [x_data, y_data], z_data, p0=[0.0, 0.0, 0.0], maxfev=int(1e6))
# Z = func(np.array([X, Y]), *fittedParameters)
#
# ax.plot(x_data, y_data, 'o')
#
# ax.set_title('Contour Plot') # add a title for contour plot
# ax.set_xlabel('X Data') # X axis data label
# ax.set_ylabel('Y Data') # Y axis data label
#
# CS = plt.contour(X, Y, Z, 16, colors='k')
# plt.clabel(CS, inline=1, fontsize=10) # labels for contours
#
# plt.show()
#
# graphWidth = 800 # units are pixels
# graphHeight = 600 # units are pixels
#
# # 3D contour plot lines
# numberOfContourLines = 16
#
#
# def SurfacePlot(func, data, fittedParameters):
#     f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
#
#     plt.grid(True)
#     axes = Axes3D(f)
#
#     x_data = data[0]
#     y_data = data[1]
#     z_data = data[2]
#
#     xModel = np.linspace(min(x_data), max(x_data), 20)
#     yModel = np.linspace(min(y_data), max(y_data), 20)
#     X, Y = np.meshgrid(xModel, yModel)
#
#     Z = func(np.array([X, Y]), *fittedParameters)
#
#     axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)
#
#     axes.scatter(x_data, y_data, z_data) # show data along with plotted surface
#
#     axes.set_title('Surface Plot (click-drag with mouse)') # add a title for surface plot
#     axes.set_xlabel('X Data') # X axis data label
#     axes.set_ylabel('Y Data') # Y axis data label
#     axes.set_zlabel('Z Data') # Z axis data label
#
#     plt.show()
#     plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems
#
#
# def ContourPlot(func, data, fittedParameters):
#     f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
#     axes = f.add_subplot(111)
#
#     x_data = data[0]
#     y_data = data[1]
#     z_data = data[2]
#
#     xModel = np.linspace(min(x_data), max(x_data), 20)
#     yModel = np.linspace(min(y_data), max(y_data), 20)
#     X, Y = np.meshgrid(xModel, yModel)
#
#     Z = func(np.array([X, Y]), *fittedParameters)
#
#     axes.plot(x_data, y_data, 'o')
#
#     axes.set_title('Contour Plot') # add a title for contour plot
#     axes.set_xlabel('X Data') # X axis data label
#     axes.set_ylabel('Y Data') # Y axis data label
#
#     CS = plt.contour(X, Y, Z, numberOfContourLines, colors='k')
#     plt.clabel(CS, inline=1, fontsize=10) # labels for contours
#
#     plt.show()
#     plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems
#
#
# def ScatterPlot(data):
#     f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
#
#     plt.grid(True)
#     axes = Axes3D(f)
#     x_data = data[0]
#     y_data = data[1]
#     z_data = data[2]
#
#     axes.scatter(x_data, y_data, z_data)
#
#     axes.set_title('Scatter Plot (click-drag with mouse)')
#     axes.set_xlabel('X Data')
#     axes.set_ylabel('Y Data')
#     axes.set_zlabel('Z Data')
#
#     plt.show()
#     plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems
#
#
# def func(data, a, alpha, beta):
#     x = data[0]
#     y = data[1]
#     return a * (x**alpha) * (y**beta)
#
#
# data = [x_data, y_data, z_data]
# initialParameters = [1.0, 1.0, 1.0]
# fittedParameters, pcov = scipy.optimize.curve_fit(func, [x_data, y_data], z_data, p0=initialParameters)
# ScatterPlot(data)
# SurfacePlot(func, data, fittedParameters)
# ContourPlot(func, data, fittedParameters)
#
# print('fitted prameters', fittedParameters)
#
# modelPredictions = func(data, *fittedParameters)
#
# absError = modelPredictions - z_data
#
# SE = np.square(absError)  # squared errors
# MSE = np.mean(SE)  # mean squared errors
# RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
# Rsquared = 1.0 - (np.var(absError) / np.var(z_data))
# print('RMSE:', RMSE)
# print('R-squared:', Rsquared)
