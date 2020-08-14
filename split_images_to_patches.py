PATCH_SIZE = 32
STRIDE = 14
FACTOR = 2

def image_split(path):
    
    x_train = []
    y_train = []
    for i, file in enumerate(os.listdir(path)):
        
        # read the file using cv2
        hr = cv2.imread(path + '/' + file)
        
        # change the image color channel to YCrCb
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)
        
        # find the old and new image dimensions
        h, w, c = hr.shape
        
        # degrade the images by downsizing and upsizing
        new_h = int(h / FACTOR)
        new_w = int(h / FACTOR) 
        lr = cv2.resize(hr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        lr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # number of stride steps
        w_steps = int((w -(PATCH_SIZE - STRIDE)) / STRIDE)
        h_steps = int((h -(PATCH_SIZE - STRIDE)) / STRIDE)
        
        #print('w: {}'.format(w))
        #print('h: {}'.format(h))
        #print('w_steps: {}'.format(w_steps))
        #print('h_steps: {}'.format(h_steps))
        
        Y_hr = np.zeros((hr.shape[0], hr.shape[1], 1), dtype=float)
        Y_hr[:, :, 0] = hr[:, :, 0].astype(float) / 255
        
        Y_lr = np.zeros((lr.shape[0], lr.shape[1], 1), dtype=float)
        Y_lr[:, :, 0] = lr[:, :, 0].astype(float) / 255
        
        for i in range(w_steps - 1):
            for j in range(h_steps - 1):
                
                hr_patch = Y_hr[i * STRIDE: i * STRIDE + PATCH_SIZE , j * STRIDE: j * STRIDE + PATCH_SIZE]
                lr_patch = Y_lr[i * STRIDE: i * STRIDE + PATCH_SIZE , j * STRIDE: j * STRIDE + PATCH_SIZE]
                
                if hr_patch.shape[0] == hr_patch.shape[1]:
                    x_train.append(hr_patch)
                    y_train.append(crop(lr_patch, 4)) 
    return x_train, y_train