import numpy as np

from function import *
from network import *
import time, os, zipfile

# --- basic constant
num_model = 13 # 11, 13, 16, 19
image_size = 128
num_cls = 200

# --- set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# --- model save
model_save_path = f'./model/VGG{num_model}/'
saving = False
saving_point = 0
if not saving:
    saving_point = 0

# --- dataset load
path = os.path.abspath('../../../dataset/classification/')
print('Data Loading ...')
z_train = zipfile.ZipFile(path + '/train.zip', 'r')
z_train_list = z_train.namelist()
train_cls = read_gt(path + '/train_gt.txt', len(z_train_list))

z_test = zipfile.ZipFile(path + '/test.zip', 'r')
z_test_list = z_test.namelist()
test_cls = read_gt(path + '/test_gt.txt', len(z_test_list))
print('Loading Finished\n')

# --- build network
print('Network Building...')
if num_model == 11:
    model = VGG11(num_cls).to(DEVICE)
elif num_model == 13:
    model = VGG13(num_cls).to(DEVICE)
elif num_model == 16:
    model = VGG16(num_cls).to(DEVICE)
elif num_model == 19:
    model = VGG19(num_cls).to(DEVICE)
else:
    import sys
    print('model number Error')
    sys.exit()
print('Build Finished\n')

if saving:
    model.load_state_dict(torch.load(model_save_path + 'model_%d.pt' % saving_point))
    model.eval()

# --- learning
loss = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
num_iter = 100000
batch_size = 8
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate
                            , weight_decay=1e-4, momentum=0.9)

start_time = time.time()
for it in range(saving_point, num_iter + 1):
    if it >= 80000 and it < 100000:
        optimizer.param_groups[0]['lr'] = 0.001

    batch_image, batch_cls = mini_batch_training_zip(z_train, z_train_list, train_cls, batch_size, image_size)
    batch_image = np.transpose(batch_image, (0, 3, 1, 2))

    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_image.astype(np.float32)).to(DEVICE))
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long).to(DEVICE)

    train_loss = loss(pred, cls_tensor)
    train_loss.backward()
    optimizer.step()

    if it % 100 == 0:
        current_time = time.time() - start_time
        print(f'it : {it}    loss : {train_loss.item():.5f}    time: {current_time:.3f}')
        start_time = time.time()

    if it % 5000 == 0:
        print('Saving Model ...')
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model.state_dict(), model_save_path + '/model_%d.pt' % it)
        print('Saving Finished\n')

    if it % 10000 == 0:
        print('Evaluating Model ...')
        model.eval()
        t1, t5 = 0, 0
        num_test_image = len(z_test_list)

        for i in range(num_test_image):
            image_temp = z_test.read(z_test_list[i])
            image_temp = cv2.imdecode(np.frombuffer(image_temp, np.uint8), 1)
            image_temp = cv2.resize(image_temp, (image_size, image_size))
            image_temp = image_temp.astype(np.float32)

            test_image = (image_temp / 255.0) * 2 - 1
            test_image = np.reshape(test_image, (1, image_size, image_size, 3))
            test_image = np.transpose(test_image, (0, 3, 1, 2))

            with torch.no_grad():
                pred = model(torch.from_numpy(test_image.astype(np.float32)).to(DEVICE))

            pred = pred.cpu().numpy()
            pred = np.reshape(pred, num_cls)

            gt = test_cls[i]

            for top in range(5):
                max_index = np.argmax(pred)
                if int(gt) == int(max_index):
                    t5 += 1

                    if top == 0:
                        t1 += 1

                pred[max_index] = -99999999
        t1 *= (100 / num_test_image); t5 *= (100 / num_test_image)
        print(f'top-1 : {t1:.4f}%    top-5 : {t5:.4f}%\n')

        f = open(f'{model_save_path}/accuracy.txt', 'a+')
        f.write(f'it : {it}    top-1 : {t1:.4f}%    top-5 : {t5:.4f}%\n')
        f.close()
        print('Evaluating Finished\n')