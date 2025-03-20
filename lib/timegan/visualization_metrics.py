'''
Description: Use PCA or tSNE for generated and original data visualization

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Based on code author: Jinsung Yoon (jsyoon0823@gmail.com)
'''


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from parameters import Paths,font1

def visualization(ori_data, generated_data, analysis, cols, name, is_save=False):
    """Using PCA or tSNE or compare for generated and original data visualization.
  
  Args:
  ----------------
    ori_data: original data
    generated_data: generated synthetic data
    analysis: ``tsne`` or ``pca`` or ``data``
    cols: columns of the data
    name: name of the plt title
    is_save: whether to save the plot
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    no, seq_len, dim = ori_data.shape
    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (prep_data_hat, np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]))
            )
    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]
    for x in analysis:
        if x == "pca":
            # PCA Analysis
            pca = PCA(n_components=2)
            pca.fit(prep_data)
            pca_results = pca.transform(prep_data)
            pca_hat_results = pca.transform(prep_data_hat)
            # Plotting
            f, ax = plt.subplots(1)
            ax.grid(False)
            plt.scatter(
                pca_results[:, 0],
                pca_results[:, 1],
                c=colors[:anal_sample_no],
                alpha=0.2,
                label="Original",
            )
            plt.scatter(
                pca_hat_results[:, 0],
                pca_hat_results[:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
            ax.legend(prop=font1)
            plt.xlabel("x-pca")
            plt.ylabel("y-pca")
            ax.xaxis.set_tick_params(width=1.5)
            ax.yaxis.set_tick_params(width=1.5)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_edgecolor('black')
            ax.spines['bottom'].set_edgecolor('black')
            ax.spines['left'].set_edgecolor('black')
            ax.spines['right'].set_edgecolor('black')
            plt.margins(0.0)
            plt.tight_layout(pad=0.1)
            if is_save:
                plt.savefig(Paths['images']+"/Ana_%s_PCA.pdf"%(name), format="pdf", bbox_inches='tight',pad_inches=0)
            plt.show()

        elif x == "tsne":
            # Do t-SNE Analysis together
            prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
            # TSNE anlaysis
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(prep_data_final)
            # Plotting
            f, ax = plt.subplots(1)
            plt.scatter(
                tsne_results[:anal_sample_no, 0],
                tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no],
                alpha=0.2,
                label="Original",
            )
            plt.scatter(
                tsne_results[anal_sample_no:, 0],
                tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
            ax.legend(prop=font1)
            # ax.legend(prop=font1, loc=4, ncol=1, frameon=False)
            plt.xlabel("x-tsne")
            plt.ylabel("y-tsne")
            ax.xaxis.set_tick_params(width=1.5)
            ax.yaxis.set_tick_params(width=1.5)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_edgecolor('black')
            ax.spines['bottom'].set_edgecolor('black')
            ax.spines['left'].set_edgecolor('black')
            ax.spines['right'].set_edgecolor('black')
            plt.margins(0.0)
            plt.tight_layout(pad=0.1)
            if is_save:
                plt.savefig(Paths['images']+"/Ana_%s_TSNE.pdf"%(name), format="pdf", bbox_inches='tight',pad_inches=0)
            plt.show()

        elif x == "data":
            idx_ = np.random.randint(0, 1000)
            for j, col in enumerate(cols):
                fig, ax = plt.subplots(1)
                x = np.arange(seq_len)
                y_real = ori_data[idx_, :, j]
                y_synthetic = generated_data[idx_, :, j]
                ax.plot(x, y_real, label='Real', color='red')
                ax.plot(x, y_synthetic, label='Synthetic', color='blue')
                ax.set_title('Synthetic vs Real on {} of {}'.format(col, name))
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend(prop=font1)
                plt.tight_layout(pad=0.1)
                if is_save:
                    plt.savefig(Paths['images'] + "/Ana_{}_Data.pdf".format(name), format="pdf", bbox_inches='tight', pad_inches=0)
                plt.show()