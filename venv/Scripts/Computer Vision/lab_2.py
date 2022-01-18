import cv2
import numpy as np


def find_template_countour(filename):
    img = cv2.imread(filename, 0)
    img_for_drawing = cv2.imread(filename)
    *t, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_for_drawing, contours, 1, (0, 255, 255), 3)
    cv2.imshow('template', img_for_drawing)

    while (True):
        if cv2.waitKey(33) == ord('q'):
            cv2.destroyWindow('template')
            break
    return contours[1]


def find_main_image_contours(filename):
    main_image = cv2.imread(filename, 0)
    copy_image = cv2.imread(filename)
    main_image = cv2.medianBlur(main_image, 3)
    ret, main_image = cv2.threshold(main_image, 50, 200, cv2.THRESH_BINARY)
    *t, contours, hierarchy = cv2.findContours(main_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return copy_image, contours


def get_keypoints(img):
    detector = cv2.SIFT_create()
    key_points, computed = detector.detectAndCompute(img, None)
    return key_points, computed


def load_data():
    import os
    apple_path = os.listdir(r'dataset\apple')
    banana_path = os.listdir(r'dataset\banana')

    apple = []

    for i in apple_path:
        img = cv2.imread(r"dataset\\apple\\" + i, 0)
        img = cv2.resize(img, (500, 500))
        apple.append(img)

    for i in banana_path:
        img = cv2.imread(r"dataset\\banana\\" + i, 0)
        img = cv2.resize(img, (500, 500))
        apple.append(img)

    apple = np.array(apple)

    return apple


def load_test_data():
    import os
    apple_path = os.listdir(r'dataset\apple_test')
    banana_path = os.listdir(r'dataset\banana_test')

    apple = []

    for i in apple_path:
        img = cv2.imread(r"dataset\\apple_test\\" + i, 0)
        img = cv2.resize(img, (500, 500))
        apple.append(img)

    apple = np.array(apple)

    banana = []

    for i in banana_path:
        img = cv2.imread(r"dataset\\banana_test\\" + i, 0)
        img = cv2.resize(img, (500, 500))
        banana.append(img)

    banana = np.array(banana)

    return apple, banana


if __name__ == '__main__':
    apple_set = load_data()

    # Алгоритм K-Means
    BOW_apple = cv2.BOWKMeansTrainer(2)
    # %%
    KP = []
    for i in apple_set:
        K, cmpt = get_keypoints(i)
        BOW_apple.add(cmpt)
        KP.append(K)

    #Число кластеров в алгоритме K-Means
    cluster = BOW_apple.cluster()
    # %%
    bow_img_DE = cv2.BOWImgDescriptorExtractor(cv2.SIFT_create(),
                                               cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE))
    bow_img_DE.setVocabulary(cluster)

    train_set = []
    for i in range(0, len(apple_set)):
        c = bow_img_DE.compute(apple_set[i], KP[i])
        train_set.append(c)

    train_set = np.array(train_set)
    #train_set = train_set.reshape(50, 2)
    train_labels = [0] * 75 + [1] * 73
    train_labels = np.array(train_labels)

    # %%
    svm = cv2.ml_SVM.create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(train_set, 0, train_labels)
    #
    predict = svm.predict(train_set)

    print('Точность: ', np.mean(predict[1] == train_labels.reshape(-1, 1)))
    print('Точность для яблок: ', np.mean(predict[1][:75] == train_labels.reshape(-1, 1)[:75]))
    print('Точность для банан: ', np.mean(predict[1][75:] == train_labels.reshape(-1, 1)[75:]))

    apple_test, banana_test = load_test_data()

    apple_test_cmp = []
    for i in range(0, len(apple_test)):
        K, cmpt = get_keypoints(apple_test[i])
        c = bow_img_DE.compute(apple_test[i], K)
        apple_test_cmp.append(c)

    banana_test_cmp = []
    for i in range(0, len(banana_test)):
        K, cmpt = get_keypoints(banana_test[i])
        c = bow_img_DE.compute(banana_test[i], K)
        banana_test_cmp.append(c)

    apple_test_cmp = np.array(apple_test_cmp)
    banana_test_cmp = np.array(banana_test_cmp)

    apple_test_cmp = apple_test_cmp.reshape((19, 2))
    banana_test_cmp = banana_test_cmp.reshape((18, 2))

    predict_apple = svm.predict(apple_test_cmp)
    predict_banana = svm.predict(banana_test_cmp)
    print('Точность тест для яблок: ', np.mean(predict_apple[1] == np.zeros(predict_apple[1].shape)))
    print('Точность тест для банан: ', np.mean(predict_banana[1] == np.ones(predict_banana[1].shape)))




