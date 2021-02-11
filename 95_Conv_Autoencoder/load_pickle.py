import pickle
from imutils import build_montages
from libs import *
from PIL import Image

tf.enable_eager_execution()

#size_print_imgs = 120
size_print_imgs = 60
rows_montage = 10
cols_montage = 30
queries_num = 10

# carregar pelo treinamento

#==================================== DATASET PATH =========================================#

#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset/24703_85_5_10_16_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset/49401_85_5_10_16_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset/98620_85_5_10_16_512_3'
dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset/196651_85_5_10_16_512_3'

#===========================================================================================#

embeddings_path = './embeddings.pickle'


def euclidean(a, b):

    return np.linalg.norm(a - b)


def perform_search(queryFeatures, index, maxResults):

    results = []

    for i in range(0, len(index["features"])):

        d = euclidean(queryFeatures, index["features"][i])
        results.append((d, i))

    results = sorted(results)[:maxResults]

    return results


img_white = np.zeros((size_print_imgs, size_print_imgs, 3))
img_white[:] = 255

test_split_size, _, _, _, _, img_size, img_depth = read_dataset_cfg_file(dataset_path)[6:]
path_tfrecords_test = glob.glob(pathname=dataset_path + slash + 'test' + slash + '*.tfrecords')
path_tfrecords_test.sort()
dataset_test = test_pipeline(_path_tfrecords=path_tfrecords_test, _img_size=img_size, _img_depth=img_depth, _batch_size=1)
images_test = []
for parsed_record in dataset_test.take(test_split_size):
    images_test.append(parsed_record[0].numpy())

embeddings = pickle.loads(open(embeddings_path, "rb").read())

queryIdxs = list(range(0, len(images_test)))
queryIdxs = np.random.choice(queryIdxs, size=queries_num, replace=False)

for i in queryIdxs:

    images = []

    if img_depth == 1:
        query = Image.fromarray(np.uint8(np.dstack([images_test[i]] * 3) * 255))
    else:
        query = Image.fromarray(np.uint8(images_test[i] * 255))

    query = query.resize([size_print_imgs, size_print_imgs])
    query = np.asarray(query)
    query = query[..., ::-1]
    images.append(query)

    for l in range(0, cols_montage - 1):
        images.append(img_white)

    queryFeatures = embeddings["features"][i]
    results = perform_search(queryFeatures, embeddings, maxResults=(rows_montage * cols_montage) + 1)

    for (dists, ids) in results:

        if ids != i:

            if img_depth == 1:
                image = Image.fromarray(np.uint8(np.dstack([images_test[ids]] * 3) * 255))
            else:
                image = Image.fromarray(np.uint8(images_test[ids] * 255))

            image = image.resize([size_print_imgs, size_print_imgs])
            image = np.asarray(image)
            image = image[..., ::-1]
            images.append(image)

    montage = build_montages(images, (size_print_imgs, size_print_imgs), (cols_montage, rows_montage + 1))[0]
    cv2.imshow("Results", montage)
    cv2.waitKey(0)
