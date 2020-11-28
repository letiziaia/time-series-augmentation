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
    beef_cnn = [0.7797619047619048, 0.7683333333333333,
                0.7892857142857143, 0.8800000000000001,
                0.7005555555555556, 0.7666666666666667,
                0.7636363636363637, 0.8333333333333334,
                0.7980952380952381, 0.7357142857142857]
    beef_cgan = [0.8, 0.8300000000000001,
                 0.8466666666666667, 0.8405555555555555,
                 0.8454545454545455, 0.8550000000000001,
                 0.9444444444444444, 0.8228571428571427,
                 0.8466666666666667, 0.7849999999999999]
    beef_jit = [0.7807142857142857, 0.9133333333333333,
                0.7619047619047619, 0.8,
                0.7899999999999999, 0.8595238095238095,
                0.7842857142857144, 0.8133333333333332,
                0.89, 0.7733333333333333]
    beef_flip = [0.6977777777777778, 0.7057142857142857,
                 0.8933333333333333, 0.75,
                 0.6805555555555556, 0.6266666666666667,
                 0.7157575757575757, 0.7083333333333334,
                 0.7880952380952382, 0.73]
    beef_perm = [0.9166666666666667, 0.7466666666666667,
                 0.7225641025641025, 0.85,
                 0.7542857142857142, 0.7771428571428571,
                 0.7733333333333333, 0.835,
                 0.775, 0.8311111111111111]
    beef_smote = [0.7264285714285714, 0.7516666666666667,
                  0.7257142857142856, 0.8555555555555555,
                  0.7476190476190476, 0.9416666666666668,
                  0.8678571428571429, 0.8333333333333334,
                  0.7404761904761904, 0.9047619047619048]
    beef = build_df(cnn=beef_cnn, jit=beef_jit, flip=beef_flip,
                    perm=beef_perm, smote=beef_smote, cgan=beef_cgan)
    # boxplot(beef, title="Beef")
    # x = Aug, y = Orig
    # test(x=beef_cgan, y=beef_cnn)
    # test(x=beef_jit, y=beef_cnn)
    # test(x=beef_flip, y=beef_cnn)
    # test(x=beef_perm, y=beef_cnn)
    # test(x=beef_smote, y=beef_cnn)
    coffee_cnn = [1.0, 0.9545454545454546,
                  1.0, 1.0,
                  1.0, 1.0,
                  1.0, 1.0,
                  1.0, 1.0]
    coffee_cgan = [1.0, 0.9545454545454546,
                   1.0, 1.0,
                   1.0, 1.0,
                   1.0, 1.0,
                   1.0, 1.0]
    coffee_jit = [1.0, 1.0,
                  1.0, 1.0,
                  1.0, 1.0,
                  1.0, 1.0,
                  1.0, 1.0]
    coffee_flip = [0.8444444444444444, 0.8888888888888888,
                   0.8512820512820514, 0.9615384615384616,
                   0.8663101604278074, 0.8636363636363636,
                   1.0, 0.7410256410256411,
                   0.78125, 0.9615384615384616]
    coffee_perm = [0.9411764705882353, 1.0,
                   1.0, 1.0,
                   1.0, 1.0,
                   1.0, 1.0,
                   1.0, 0.96875]
    coffee_smote = [1.0, 1.0,
                    1.0, 1.0,
                    1.0, 1.0,
                    1.0, 0.9583333333333333,
                    1.0, 1.0]
    coffee = build_df(cnn=coffee_cnn, jit=coffee_jit, flip=coffee_flip,
                      perm=coffee_perm, smote=coffee_smote, cgan=coffee_cgan)
    # boxplot(coffee, title="Coffee")
    # x = Aug, y = Orig
    # test(x=coffee_cgan, y=coffee_cnn)
    # test(x=coffee_jit, y=coffee_cnn)
    # test(x=coffee_flip, y=coffee_cnn)
    # test(x=coffee_perm, y=coffee_cnn)
    # test(x=coffee_smote, y=coffee_cnn)
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
    freezer_perm = [0.9599167908594138, ]
    freezer_smote = [0.9875886524822695, 0.9886010140641925,
                     0.9861737327522636, 0.991534209133106,
                     0.9895199459093982, 0.9761570827489481,
                     0.9855144144015441, 0.984284262199494,
                     0.9825326079832208, 0.9849223686906095]
    freezer = build_df(cnn=freezer_cnn, jit=freezer_jit, flip=freezer_flip,
                       perm=freezer_perm, smote=freezer_smote, cgan=freezer_cgan)
    # boxplot(freezer, title="Freezer")
    # x = Aug, y = Orig
    # test(x=freezer_cgan, y=freezer_cnn)
    # test(x=freezer_jit, y=freezer_cnn)
    # test(x=freezer_flip, y=freezer_cnn)
    # test(x=freezer_perm, y=freezer_cnn)
    # test(x=freezer_smote, y=freezer_cnn)

    gunpoint_cnn = [0.9464793741109531, 0.9198079658605974,
                    0.9532361308677098, 0.9464793741109531,
                    0.9458013906222142, 0.9500379939209727,
                    0.9666666666666667, 0.9334415584415585,
                    0.9267857142857143, 0.96]
    gunpoint_cgan = []
    gunpoint_jit = [0.9668856048166393, 0.9660714285714286,
                    0.9587609970674487, 0.9802631578947368,
                    0.9665718349928876, 0.9545940170940171,
                    0.9583931133428981, 0.9401854162952398,
                    0.9794520547945205, 0.9517857142857142]
    gunpoint_flip = []
    gunpoint_permute = []
    gunpoint_smote = []
    gunpoint = build_df(cnn=gunpoint_cnn, jit=gunpoint_jit, flip=gunpoint_flip,
                        perm=gunpoint_permute, smote=gunpoint_smote, cgan=gunpoint_cgan)
    # boxplot(gunpoint, title="Gunpoint")
    # x = Aug, y = Orig
    # test(x=gunpoint_cgan, y=gunpoint_cnn)
    # test(x=gunpoint_jit, y=gunpoint_cnn)
    # test(x=gunpoint_flip, y=gunpoint_cnn)
    # test(x=gunpoint_perm, y=gunpoint_cnn)
    # test(x=gunpoint_smote, y=gunpoint_cnn)
