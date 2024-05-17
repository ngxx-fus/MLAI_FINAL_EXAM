#---------------------- local func and var ----------------------
from Dataset_Def  import *
from PSPNet_def   import *
from Train_Model  import *
#---------------------- WEIGHTS PATH DEFINATION ----------------------
weights_path = r"./Model_Weights"


#---------------------- LOAD WEIGHTS ----------------------
PATH = weights_path + r"/29th_modelPSPNet.pth"
my_PSPnet_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

#---------------------- TEST MODEL 1 DEFINATION ----------------------
my_PSPnet_model.eval()

for id in range(test_dataset.__len__()):
  with torch.no_grad():
        iou_meter.reset()
        x, y = test_dataset.__getitem__(id)
        n = x.shape[0]
        y_predict = my_PSPnet_model(x.unsqueeze(0).to(device)).argmax(dim=1).squeeze().numpy()
        color_mask_predict = unorm(x).permute(1, 2, 0)
        plt.subplot(1,2,1)
        plt.imshow(color_mask_predict)
        for i in range(y_predict.shape[0]):
            for j in range(y_predict.shape[1]):
                if y_predict[i,j] == 2:
                    color_mask_predict[i,j, 1] += 0.32
                elif y_predict[i,j] == 3:
                    color_mask_predict[i,j, 2] += 0.32
                elif y_predict[i,j] == 4:
                    color_mask_predict[i,j, 0] += 0.32

        plt.subplot(1,2,2)
        plt.imshow(color_mask_predict)
        plt.show()

# print(iou_meter.avg)

 