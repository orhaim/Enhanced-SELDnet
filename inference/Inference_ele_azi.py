from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from keras_self_attention import SeqSelfAttention
import numpy as np
import matplotlib.pyplot as plt


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

def cart2sph(x, y, z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

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

colors = {0: ['c', "."], 1: ['b', "v"], 2: ['g', "^"], 3: ['r', "<"], 4: ['m', ">"], 5: ['yellow', "s"], 6: ['k', "P"], 7: ['violet', "D"], 8: ['peru', "*"], 9: ['silver', "X"], 10: ['palegreen', "d"]}

figx = plt.figure()
ax = figx.add_subplot(111)
ax.set_title('X-axis DOA Predicted')
ax.set_ylim(-1, 1)
figx_labels = plt.figure()
ax_labels = figx_labels.add_subplot(111)
ax_labels.set_title('X-axis DOA Labels')
ax_labels.set_ylim(-1, 1)
ax.set_xticklabels([])
ax_labels.set_xticklabels([])

figy = plt.figure()
ay = figy.add_subplot(111)
ay.set_title('Y-axis DOA Predicted')
ay.set_ylim(-1, 1)
figy_labels = plt.figure()
ay_labels = figy_labels.add_subplot(111)
ay_labels.set_title('Y-axis DOA Labels')
ay_labels.set_ylim(-1, 1)
ay.set_xticklabels([])
ay_labels.set_xticklabels([])

figz = plt.figure()
az = figz.add_subplot(111)
az.set_title('Z-axis DOA Predicted')
az.set_ylim(-1, 1)
figz_labels = plt.figure()
az_labels = figz_labels.add_subplot(111)
az_labels.set_title('Z-axis DOA Labels')
az_labels.set_ylim(-1, 1)
az.set_xticklabels([])
az_labels.set_xticklabels([])

fig_sed = plt.figure()
a_sed = fig_sed.add_subplot(111)
a_sed.set_title('SED Predicted')
a_sed.set_ylim(-1, 11)
fig_sed_labels = plt.figure()
a_sed_labels = fig_sed_labels.add_subplot(111)
a_sed_labels.set_title('SED Labels')
a_sed_labels.set_ylim(-1, 11)
a_sed.set_xticklabels([])
a_sed_labels.set_xticklabels([])

low, high = 3000, 4000

for i, col in enumerate(np.transpose(sed)):
    for j, row in enumerate(col[low:high]):
        if row == 1:
            a_sed.scatter(j, i, c=colors[i][0], s=1)
            x, y, z = doa[j, i], doa[j, i + 11], doa[j, i + 22]
            ax.scatter(j, x, c=colors[i][0], s=1)
            ay.scatter(j, y, c=colors[i][0], s=1)
            az.scatter(j, z, c=colors[i][0], s=1)
            # x_label, y_label, z_labels = sph2cart(label_doa[j, i]+(5*180/np.pi), label_doa[j, i + 11]+(5*180/np.pi), 1)
            x_label, y_label, z_label = sph2cart(label_doa[j, i], label_doa[j, i + 11], 1)
            # if x_label < 0:
            #     x_label, y_label, z_labels = sph2cart(label_doa[j, i]+(5*180/np.pi), label_doa[j, i + 11]+(5*180/np.pi), 1)
            if -0.55 <= x_label <= -0.45:
                # x_label += 0.5
                x_label, y_label, _ = sph2cart(label_doa[j, i]+(5*180/np.pi), label_doa[j, i + 11]+(5*180/np.pi), 1)
                # _, _, z_label = sph2cart(label_doa[j, i]+(17.7*180/np.pi), label_doa[j, i + 11]+(17.7*180/np.pi), 1)
                z_label -= 0.86
            ax_labels.scatter(j, x_label, c=colors[i][0], s=1, alpha=0.3)
            ay_labels.scatter(j, y_label, c=colors[i][0], s=1, alpha=0.3)
            az_labels.scatter(j, z_label, c=colors[i][0], s=1, alpha=0.3)

for i, col in enumerate(np.transpose(label_sed)):
    for j, row in enumerate(col[low:high]):
        if row == 1:
            a_sed_labels.scatter(j, i, c=colors[i][0], marker="+", s=1)

plt.show()




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('DOA & SED Predicted')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
# fig_labels = plt.figure()
# ax_labels = fig_labels.add_subplot(111, projection='3d')
# ax_labels.set_title('DOA & SED GT')
# ax_labels.set_xlim(-1, 1)
# ax_labels.set_ylim(-1, 1)
# ax_labels.set_zlim(-1, 1)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.set_title('DOA Predicted - SED GT')
# ax1.set_xlim(-1, 1)
# ax1.set_ylim(-1, 1)
# ax1.set_zlim(-1, 1)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.set_title('DOA GT - SED Predicted')
# ax2.set_xlim(-1, 1)
# ax2.set_ylim(-1, 1)
# ax2.set_zlim(-1, 1)
# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111, projection='3d')
# ax3.set_xlim(-0.6, 0.6)
# ax3.set_ylim(-0.6, 0.6)
# ax3.set_zlim(-0.6, 0.6)
# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111, projection='3d')
# ax4.set_xlim(-0.6, 0.6)
# ax4.set_ylim(-0.6, 0.6)
# ax4.set_zlim(-0.6, 0.6)
# fig5 = plt.figure()
# ax5 = fig5.add_subplot(111, projection='3d')
# ax5.set_xlim(-0.6, 0.6)
# ax5.set_ylim(-0.6, 0.6)
# ax5.set_zlim(-0.6, 0.6)
# fig6 = plt.figure()
# ax6 = fig6.add_subplot(111, projection='3d')
# ax6.set_xlim(-0.6, 0.6)
# ax6.set_ylim(-0.6, 0.6)
# ax6.set_zlim(-0.6, 0.6)
# fig7 = plt.figure()
# ax7 = fig7.add_subplot(111, projection='3d')
# ax7.set_xlim(-0.6, 0.6)
# ax7.set_ylim(-0.6, 0.6)
# ax7.set_zlim(-0.6, 0.6)
# fig8 = plt.figure()
# ax8 = fig8.add_subplot(111, projection='3d')
# ax8.set_xlim(-0.6, 0.6)
# ax8.set_ylim(-0.6, 0.6)
# ax8.set_zlim(-0.6, 0.6)
# fig9 = plt.figure()
# ax9 = fig9.add_subplot(111, projection='3d')
# ax9.set_xlim(-0.6, 0.6)
# ax9.set_ylim(-0.6, 0.6)
# ax9.set_zlim(-0.6, 0.6)
# fig10 = plt.figure()
# ax10 = fig10.add_subplot(111, projection='3d')
# ax10.set_xlim(-0.6, 0.6)
# ax10.set_ylim(-0.6, 0.6)
# ax10.set_zlim(-0.6, 0.6)
# fig11 = plt.figure()
# ax11 = fig11.add_subplot(111, projection='3d')
# ax11.set_xlim(-0.6, 0.6)
# ax11.set_ylim(-0.6, 0.6)
# ax11.set_zlim(-0.6, 0.6)
# # fig12 = plt.figure()
# # ax12 = fig12.add_subplot(111, projection='3d')
# # ax12.set_xlim(-0.6, 0.6)
# # ax12.set_ylim(-0.6, 0.6)
# # ax12.set_zlim(-0.6, 0.6)
# # fig13 = plt.figure()
# # ax13 = fig13.add_subplot(111, projection='3d')
# # ax13.set_xlim(-0.6, 0.6)
# # ax13.set_ylim(-0.6, 0.6)
# # ax13.set_zlim(-0.6, 0.6)
# # fig14 = plt.figure()
# # ax14 = fig14.add_subplot(111, projection='3d')
# # ax14.set_xlim(-0.6, 0.6)
# # ax14.set_ylim(-0.6, 0.6)
# # ax14.set_zlim(-0.6, 0.6)
# # fig15 = plt.figure()
# # ax15 = fig15.add_subplot(111, projection='3d')
# # ax15.set_xlim(-0.6, 0.6)
# # ax15.set_ylim(-0.6, 0.6)
# # ax15.set_zlim(-0.6, 0.6)
# # fig16 = plt.figure()
# # ax16 = fig16.add_subplot(111, projection='3d')
# # ax16.set_xlim(-0.6, 0.6)
# # ax16.set_ylim(-0.6, 0.6)
# # ax16.set_zlim(-0.6, 0.6)
# # fig17 = plt.figure()
# # ax17 = fig17.add_subplot(111, projection='3d')
# # ax17.set_xlim(-0.6, 0.6)
# # ax17.set_ylim(-0.6, 0.6)
# # ax17.set_zlim(-0.6, 0.6)
# # fig18 = plt.figure()
# # ax18 = fig18.add_subplot(111, projection='3d')
# # ax18.set_xlim(-0.6, 0.6)
# # ax18.set_ylim(-0.6, 0.6)
# # ax18.set_zlim(-0.6, 0.6)
# # fig19 = plt.figure()
# # ax19 = fig19.add_subplot(111, projection='3d')
# # ax19.set_xlim(-0.6, 0.6)
# # ax19.set_ylim(-0.6, 0.6)
# # ax19.set_zlim(-0.6, 0.6)
# # fig20 = plt.figure()
# # ax20 = fig20.add_subplot(111, projection='3d')
# # ax20.set_xlim(-0.6, 0.6)
# # ax20.set_ylim(-0.6, 0.6)
# # ax20.set_zlim(-0.6, 0.6)
# # fig21 = plt.figure()
# # ax21 = fig21.add_subplot(111, projection='3d')
# # ax21.set_xlim(-0.6, 0.6)
# # ax21.set_ylim(-0.6, 0.6)
# # ax21.set_zlim(-0.6, 0.6)
# # fig22 = plt.figure()
# # ax22 = fig22.add_subplot(111, projection='3d')
# # ax22.set_xlim(-0.6, 0.6)
# # ax22.set_ylim(-0.6, 0.6)
# # ax22.set_zlim(-0.6, 0.6)
# # fig23 = plt.figure()
# # ax23 = fig23.add_subplot(111, projection='3d')
# # ax23.set_xlim(-0.6, 0.6)
# # ax23.set_ylim(-0.6, 0.6)
# # ax23.set_zlim(-0.6, 0.6)
# # fig24 = plt.figure()
# # ax24 = fig24.add_subplot(111, projection='3d')
# # ax24.set_xlim(-0.6, 0.6)
# # ax24.set_ylim(-0.6, 0.6)
# # ax24.set_zlim(-0.6, 0.6)
#
# colors = {0: ['c', ".", ax1], 1: ['b', "v", ax2], 2: ['g', "^", ax3], 3: ['r', "<", ax4], 4: ['m', ">", ax5], 5: ['yellow', "s", ax6], 6: ['k', "P", ax7], 7: ['violet', "D", ax8], 8: ['peru', "*", ax9], 9: ['silver', "X", ax10], 10: ['palegreen', "d", ax11]}
# # for i, row in enumerate(sed):
# #     for j, col in enumerate(row):
# #         if col == 1:
# #             x, y, z = doa[i, j], doa[i, j+11], doa[i, j+22]
# #             ax.scatter(x, y, z, c=colors[j][0], s=2)#, marker=colors[j][1])
#             # if j == 0:
#             #     ax1.scatter(x, y, z, c='b', s=2)
#             # if j == 1:
#             #     ax2.scatter(x, y, z, c='b', s=2)
#             # if j == 2:
#             #     ax3.scatter(x, y, z, c='b', s=2)
#             # if j == 3:
#             #     ax4.scatter(x, y, z, c='b', s=2)
#             # if j == 4:
#             #     ax5.scatter(x, y, z, c='b', s=2)
#             # if j == 5:
#             #     ax6.scatter(x, y, z, c='b', s=2)
#             # if j == 6:
#             #     ax7.scatter(x, y, z, c='b', s=2)
#             # if j == 7:
#             #     ax8.scatter(x, y, z, c='b', s=2)
#             # if j == 8:
#             #     ax9.scatter(x, y, z, c='b', s=2)
#             # if j == 9:
#             #     ax10.scatter(x, y, z, c='b', s=2)
#             # if j == 10:
#             #     ax11.scatter(x, y, z, c='b', s=2)
#
# for i, col in enumerate(np.transpose(sed)):
#     for j, row in enumerate(col):
#         if row == 1:
#             x, y, z = sph2cart(label_doa[j, i], label_doa[j, i+11], 1)
#             ax2.scatter(x, y, z, c=colors[i][0], s=2)
#
# for i, col in enumerate(np.transpose(label_sed)):
#     for j, row in enumerate(col):
#         if row == 1:
#             x, y, z = sph2cart(label_doa[j, i], label_doa[j, i+11], 1)
#             ax_labels.scatter(x, y, z, c=colors[i][0], s=2)    # , marker=colors[j][1])
# #             if j == 0:
# #                 ax1.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 1:
# #                 ax2.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 2:
# #                 ax3.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 3:
# #                 ax4.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 4:
# #                 ax5.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 5:
# #                 ax6.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 6:
# #                 ax7.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 7:
# #                 ax8.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 8:
# #                 ax9.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 9:
# #                 ax10.scatter(x, y, z, c='y', s=2, alpha=0.8)
# #             if j == 10:
# #                 ax11.scatter(x, y, z, c='y', s=2, alpha=0.8)
#
# x_list, y_list, z_list = [], [], []
# for i, col in enumerate(np.transpose(sed)):
#     for j, row in enumerate(col):
#         if row == 1:
#             x, y, z = doa[j, i], doa[j, i+11], doa[j, i+22]
#             x_list.append(x)
#             y_list.append(y)
#             z_list.append(z)
#             if j + 1 == len(col):
#                 x_avg, y_avg, z_avg = np.average(x_list), np.average(y_list), np.average(z_list)
#                 ax.scatter(x_avg, y_avg, z_avg, c=colors[i][0], s=2)  # , marker=colors[j][1])
#                 x_list, y_list, z_list = [], [], []
#         if row == 0:
#             if len(x_list) != 0:
#                 x_avg, y_avg, z_avg = np.average(x_list), np.average(y_list), np.average(z_list)
#                 ax.scatter(x_avg, y_avg, z_avg, c=colors[i][0], s=2)#, marker=colors[j][1])
#                 x_list, y_list, z_list = [], [], []
#
# for i, col in enumerate(np.transpose(label_sed)):
#     for j, row in enumerate(col):
#         if row == 1:
#             x, y, z = doa[j, i], doa[j, i + 11], doa[j, i + 22]
#             x_list.append(x)
#             y_list.append(y)
#             z_list.append(z)
#             if j + 1 == len(col):
#                 x_avg, y_avg, z_avg = np.average(x_list), np.average(y_list), np.average(z_list)
#                 ax1.scatter(x_avg, y_avg, z_avg, c=colors[i][0], s=2)  # , marker=colors[j][1])
#                 x_list, y_list, z_list = [], [], []
#         if row == 0:
#             if len(x_list) != 0:
#                 x_avg, y_avg, z_avg = np.average(x_list), np.average(y_list), np.average(z_list)
#                 ax1.scatter(x_avg, y_avg, z_avg, c=colors[i][0], s=2)  # , marker=colors[j][1])
#                 x_list, y_list, z_list = [], [], []
#
# plt.show()
