import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


def build_df(cnn, cgan, jit, flip, smote, perm):
    return pd.DataFrame({"CNN (No Augmentation)": cnn,
                         "CNN (Orig. + Jitter)": jit,
                         "CNN (Orig. + Flip)": flip,
                         "CNN (Orig. + Permute)": perm,
                         "CNN (Orig. + AVG_TS_SMOTE)": smote,
                         "CNN (Orig + C-GAN)": cgan})


def boxplot(df, title=""):
    title = "Balanced Accuracy Score (" + title + ")"
    sns.boxplot(data=df, orient="h", palette="pastel")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def test(x, y):
    """
    Tests whether the distribution of the differences x - y is symmetric about zero
    :param x: 1d np.array or list, potential improvement
    :param y: 1d np.array or list, baseline
    :return: None
    """
    print("x median: ", np.sort(np.array(x))[4:6].mean())
    print("x mean: ", np.array(x).mean())
    print("x std: ", np.array(x).std())
    print("y median: ", np.sort(np.array(y))[4:6].mean())
    print("y mean: ", np.array(y).mean())
    print("y std: ", np.array(y).std())
    # Two-sided test:
    # Null hypothesis: the median of the differences is zero
    # Alternative hypothesis: the median is different from zero
    print("Two-sided test")
    stat, p = wilcoxon(x, y, alternative="two-sided")
    print(p)
    if p < 0.05:
        print("We reject the null hypothesis at a confidence level of 0.05%")
    else:
        print("We cannot reject the null hypothesis")
    # One-sided test:
    # Null hypothesis: the median is negative
    # Alternative hypothesis: the median is positive
    print("One-sided test")
    stat, p = wilcoxon(x, y, alternative="greater")
    print(p)
    if p < 0.05:
        print("We reject the null hypothesis at a confidence level of 0.05%")
    else:
        print("We cannot reject the null hypothesis")


if __name__ == '__main__':
    # balanced accuracy
    # BEEF --> DONE
    # beef_cnn = [0.7797619047619048, 0.7683333333333333,
    #             0.7892857142857143, 0.8800000000000001,
    #             0.7005555555555556, 0.7666666666666667,
    #             0.7636363636363637, 0.8333333333333334,
    #             0.7980952380952381, 0.7357142857142857]
    # beef_cgan = [0.8, 0.8300000000000001,
    #              0.8466666666666667, 0.8405555555555555,
    #              0.8454545454545455, 0.8550000000000001,
    #              0.9444444444444444, 0.8228571428571427,
    #              0.8466666666666667, 0.7849999999999999]
    # beef_jit = [0.7807142857142857, 0.9133333333333333,
    #             0.7619047619047619, 0.8,
    #             0.7899999999999999, 0.8595238095238095,
    #             0.7842857142857144, 0.8133333333333332,
    #             0.89, 0.7733333333333333]
    # beef_flip = [0.6977777777777778, 0.7057142857142857,
    #              0.8933333333333333, 0.75,
    #              0.6805555555555556, 0.6266666666666667,
    #              0.7157575757575757, 0.7083333333333334,
    #              0.7880952380952382, 0.73]
    # beef_perm = [0.9166666666666667, 0.7466666666666667,
    #              0.7225641025641025, 0.85,
    #              0.7542857142857142, 0.7771428571428571,
    #              0.7733333333333333, 0.835,
    #              0.775, 0.8311111111111111]
    # beef_smote = [0.7264285714285714, 0.7516666666666667,
    #               0.7257142857142856, 0.8555555555555555,
    #               0.7476190476190476, 0.9416666666666668,
    #               0.8678571428571429, 0.8333333333333334,
    #               0.7404761904761904, 0.9047619047619048]
    # beef = build_df(cnn=beef_cnn, jit=beef_jit, flip=beef_flip,
    #                 perm=beef_perm, smote=beef_smote, cgan=beef_cgan)
    # boxplot(beef, title="Beef")
    # x = Aug, y = Orig
    # test(x=beef_cgan, y=beef_cnn)
    # test(x=beef_jit, y=beef_cnn)
    # test(x=beef_flip, y=beef_cnn)
    # test(x=beef_perm, y=beef_cnn)
    # test(x=beef_smote, y=beef_cnn)
    # COFFEE --> DONE
    # coffee_cnn = [1.0, 0.9545454545454546,
    #               1.0, 1.0,
    #               1.0, 1.0,
    #               1.0, 1.0,
    #               1.0, 1.0]
    # coffee_cgan = [1.0, 0.9545454545454546,
    #                1.0, 1.0,
    #                1.0, 1.0,
    #                1.0, 1.0,
    #                1.0, 1.0]
    # coffee_jit = [1.0, 1.0,
    #               1.0, 1.0,
    #               1.0, 1.0,
    #               1.0, 1.0,
    #               1.0, 1.0]
    # coffee_flip = [0.8444444444444444, 0.8888888888888888,
    #                0.8512820512820514, 0.9615384615384616,
    #                0.8663101604278074, 0.8636363636363636,
    #                1.0, 0.7410256410256411,
    #                0.78125, 0.9615384615384616]
    # coffee_perm = [0.9411764705882353, 1.0,
    #                1.0, 1.0,
    #                1.0, 1.0,
    #                1.0, 1.0,
    #                1.0, 0.96875]
    # coffee_smote = [1.0, 1.0,
    #                 1.0, 1.0,
    #                 1.0, 1.0,
    #                 1.0, 0.9583333333333333,
    #                 1.0, 1.0]
    # coffee = build_df(cnn=coffee_cnn, jit=coffee_jit, flip=coffee_flip,
    #                   perm=coffee_perm, smote=coffee_smote, cgan=coffee_cgan)
    # boxplot(coffee, title="Coffee")
    # x = Aug, y = Orig
    # test(x=coffee_cgan, y=coffee_cnn)
    # test(x=coffee_jit, y=coffee_cnn)
    # test(x=coffee_flip, y=coffee_cnn)
    # test(x=coffee_perm, y=coffee_cnn)
    # test(x=coffee_smote, y=coffee_cnn)
    # FREEZER --> CGAN MISSING
    freezer_cnn = [0.9866666666666667, 0.9868942080378251,
                   0.9590222861250899, 0.9830028328611897,
                   0.9835960591133004, 0.984561403508772,
                   0.9903632265454679, 0.9862668769094314,
                   0.984308753138471, 0.9874184288401051]
    freezer_cgan = [0.9887719298245614, 0.9738330975954739,
                    0.9760101507834829, ]
    freezer_jit = [0.9905817882560832, 0.9862258640984114,
                   0.984199930661708, 0.9889334658161363,
                   0.9895104895104896, 0.9860945642521851,
                   0.9890894316950656, 0.9843314763231198,
                   0.9576315509538331, 0.9855989945909982]
    freezer_flip = [0.9761575673807878, 0.9820931890886742,
                    0.9849246231155779, 0.9854388585731869,
                    0.9835419399827441, 0.9860485662955345,
                    0.985032074126871, 0.9771448663853728,
                    0.9800947068952609, 0.9827838827838828]
    freezer_perm = [0.9599167908594138, 0.9518510893164633,
                    0.9614816390691299, 0.9527848255864346,
                    0.9530677493321986, 0.9628325568357324,
                    0.9730835602273752, 0.9585233745107686,
                    0.9544115409107496, 0.9635480968629794]
    freezer_smote = [0.9875886524822695, 0.9886010140641925,
                     0.9861737327522636, 0.991534209133106,
                     0.9895199459093982, 0.9761570827489481,
                     0.9855144144015441, 0.984284262199494,
                     0.9825326079832208, 0.9849223686906095]
    # freezer = build_df(cnn=freezer_cnn, jit=freezer_jit, flip=freezer_flip,
    #                   perm=freezer_perm, smote=freezer_smote, cgan=freezer_cgan)
    # boxplot(freezer, title="Freezer")
    # x = Aug, y = Orig
    # test(x=freezer_cgan, y=freezer_cnn)
    # test(x=freezer_jit, y=freezer_cnn)
    # test(x=freezer_flip, y=freezer_cnn)
    # test(x=freezer_perm, y=freezer_cnn)
    # test(x=freezer_smote, y=freezer_cnn)
    # GUNPOINT --> DONE
    # gunpoint_cnn = [0.9464793741109531, 0.9198079658605974,
    #                 0.9532361308677098, 0.9464793741109531,
    #                 0.9458013906222142, 0.9500379939209727,
    #                 0.9666666666666667, 0.9334415584415585,
    #                 0.9267857142857143, 0.96]
    # gunpoint_cgan = [0.9783798576902025, 0.9586688137412775,
    #                  0.9366883116883117, 0.9538338373954812,
    #                  0.98, 0.9416666666666667,
    #                  0.9145299145299145, 0.9393669985775248,
    #                  0.9412238325281803, 0.9145299145299145]
    # gunpoint_jit = [0.9668856048166393, 0.9660714285714286,
    #                 0.9587609970674487, 0.9802631578947368,
    #                 0.9665718349928876, 0.9545940170940171,
    #                 0.9583931133428981, 0.9401854162952398,
    #                 0.9794520547945205, 0.9517857142857142]
    # gunpoint_flip = [0.9337606837606838, 0.9920634920634921,
    #                  0.9732572651096452, 0.9653679653679654,
    #                  0.9671766589574808, 0.9472276698163665,
    #                  0.926731078904992, 0.9725441255125691,
    #                  0.9338563023711892, 0.9866666666666667]
    # gunpoint_perm = [0.9596157267390144, 0.9657534246575342,
    #                  0.9527664116705212, 0.9821428571428572,
    #                  0.993421052631579, 0.9804519368723099,
    #                  0.9878048780487805, 0.9484702093397746,
    #                  0.9733333333333334, 0.9823529411764707]
    # gunpoint_smote = [0.941611695489392, 0.9664650418075076,
    #                   0.9473214285714286, 0.9466666666666667,
    #                   0.9165324745034891, 0.9134495641344956,
    #                   0.9487087517934003, 0.9563927351195828,
    #                   0.9069264069264069, 0.92]
    # gunpoint = build_df(cnn=gunpoint_cnn, jit=gunpoint_jit, flip=gunpoint_flip,
    #                    perm=gunpoint_perm, smote=gunpoint_smote, cgan=gunpoint_cgan)
    # boxplot(gunpoint, title="Gunpoint")
    # x = Aug, y = Orig
    # test(x=gunpoint_cgan, y=gunpoint_cnn)
    # test(x=gunpoint_jit, y=gunpoint_cnn)
    # test(x=gunpoint_flip, y=gunpoint_cnn)
    # test(x=gunpoint_perm, y=gunpoint_cnn)
    # test(x=gunpoint_smote, y=gunpoint_cnn)
    # ECG200 --> DONE
    # ecg_cnn = [0.8194614443084456, 0.7491319444444444,
    #            0.7708333333333333, 0.7664092664092663,
    #            0.7925347222222222, 0.7666666666666666,
    #            0.760438197602315, 0.7883310719131615,
    #            0.7875, 0.764367816091954]
    # ecg_cgan = [0.823076923076923, 0.8062211013030685,
    #             0.8376736111111112, 0.8341346153846154,
    #             0.7885572139303483, 0.8491632745364088,
    #             0.8168094655242758, 0.8057309373482273,
    #             0.8307620383356709, 0.840213695968917]
    # ecg_jit = [0.8516042780748663, 0.8131868131868132,
    #            0.8646010186757216, 0.7896213183730716,
    #            0.817062818336163, 0.788783355947535,
    #            0.7986886838271006, 0.7958404074702886,
    #            0.8338999514327343, 0.7958333333333334]
    # ecg_flip = [0.7932263814616756, 0.8833333333333333,
    #             0.8179907524169819, 0.7583333333333333,
    #             0.8041666666666667, 0.8570782451379466,
    #             0.787001287001287, 0.8601458601458601,
    #             0.7692307692307692, 0.872003618272275]
    # ecg_perm = [0.8683409436834094, 0.8386958386958387,
    #             0.8901098901098901, 0.8603565365025467,
    #             0.8186341022161918, 0.8581112669471715,
    #             0.8541666666666667, 0.8483516483516484,
    #             0.8602611179110566, 0.8941655359565808]
    # ecg_smote = [0.841055341055341, 0.821658615136876,
    #              0.8547619047619048, 0.8390079865489701,
    #              0.8785714285714286, 0.8676470588235294,
    #              0.8726655348047538, 0.8214285714285714,
    #              0.8542673107890499, 0.8715277777777778]
    # ecg = build_df(cnn=ecg_cnn, jit=ecg_jit, flip=ecg_flip,
    #                perm=ecg_perm, smote=ecg_smote, cgan=ecg_cgan)
    # boxplot(ecg, title="ECG200")
    # x = Aug, y = Orig
    # test(x=ecg_cgan, y=ecg_cnn)
    # test(x=ecg_jit, y=ecg_cnn)
    # test(x=ecg_flip, y=ecg_cnn)
    # test(x=ecg_perm, y=ecg_cnn)
    # test(x=ecg_smote, y=ecg_cnn)
    # INSECTEPG -->
    # ins_cnn = []
    # ins_cgan = []
    # ins_jit = []
    # ins_flip = []
    # ins_perm = []
    # ins_smote = []
    # ins = build_df(cnn=ins_cnn, jit=ins_jit, flip=ins_flip,
    #                perm=ins_perm, smote=ins_smote, cgan=ins_cgan)
    # boxplot(ins, title="InsectEPG")
    # x = Aug, y = Orig
    # test(x=ins_cgan, y=ins_cnn)
    # test(x=ins_jit, y=ins_cnn)
    # test(x=ins_flip, y=ins_cnn)
    # test(x=ins_perm, y=ins_cnn)
    # test(x=ins_smote, y=ins_cnn)
