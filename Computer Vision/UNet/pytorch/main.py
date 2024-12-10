from function import *
from network import *
import time, os

# --- basic constant
image_size = 128
num_cls = 21

# --- set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# --- model save
model_save_path = './model/UNet/'
image_save_path = './seg_result/UNet/'
saving = False
saving_point = 0
if not saving:
    saving_point = 0

# --- dataset load
path = os.path.abspath('../../../dataset/VOC/')
print('Data Loading ...')
train_image, train_gt = load_semantic_seg_data(path + '/train/train_img/', path + '/train/train_gt/', image_size=image_size)
test_image, test_gt = load_semantic_seg_data(path + '/test/test_img/', path + '/test/test_gt/', image_size=image_size)
print('Loading Finished\n')

# --- build network
print('Network Building...')
model = UNet(num_cls).to(DEVICE)
print('Build Finished\n')

if saving:
    model.load_state_dict(torch.load(model_save_path + 'model_%d.pt' % saving_point))
    model.eval()

# --- learning
learning_rate = 0.01
num_iter = 100000
batch_size = 8
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate
                            , weight_decay=5e-4, momentum=0.9)

start_time = time.time()
for it in range(saving_point, num_iter + 1):
    if it >= 80000 and it < 100000:
        optimizer.param_groups[0]['lr'] = 0.001

    batch_image, batch_gt = mini_batch_training(train_image, train_gt, batch_size, image_size=image_size)
    batch_image = np.transpose(batch_image, (0, 3, 1, 2))

    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_image.astype(np.float32)).to(DEVICE))

    gt_tensor = torch.tensor(batch_gt, dtype=torch.long).to(DEVICE)
    gt_tensor = torch.permute(gt_tensor, (0, 3, 1, 2)).squeeze()

    train_loss = torch.nn.functional.cross_entropy(pred, gt_tensor)
    train_loss.backward()
    optimizer.step()

    if it % 100 == 0:
        current_time = time.time() - start_time
        print(f'it : {it}   loss : {train_loss.item():.5f}  time: {current_time:.3f}')
        start_time = time.time()

    if it % 5000 == 0:
        print('Saving Model ...')
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model.state_dict(), model_save_path + '/model_%d.pt' % it)
        print('Saving Finished\n')

    if it % 10000 == 0:
        print('Evaluating Finished')
        model.eval()

        for i in range(len(test_image)):
            image_temp = test_image[i: i + 1, :, :, :].astype(np.float32)
            image_temp = (image_temp / 255.0) * 2 - 1
            image_temp = np.transpose(image_temp, (0, 3, 1, 2))

            with torch.no_grad():
                pred = model(torch.from_numpy(image_temp.astype(np.float32)).to(DEVICE))

            pred = pred.cpu().numpy()
            pred = np.argmax(pred[0, :, :, :], axis=0)
            pred = pred[:, :, np.newaxis]

            test_save = np.zeros(shape=(image_size, image_size, 3), dtype=np.uint8)
            for ic in range(len(VOC_COLORMAP)):
                code = VOC_COLORMAP[ic]
                test_save[np.where(np.all(pred == ic, axis=-1))] = code

            big_paper = np.ones(shape=(image_size, 2 * image_size, 3), dtype=np.uint8)
            big_paper[:, :image_size, :] = test_image[i: i + 1, :, :, :]
            big_paper[:, image_size:, :] = test_save

            temp = image_save_path + '%d/' %it
            if not os.path.isdir(temp):
                os.makedirs(temp)

            cv2.imwrite(temp + '%d.png' % (i), big_paper)
            print('Evaluating Finished\n')