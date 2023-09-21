import mmdet

print("ok")
# import os
# from mmdet.apis import init_detector, inference_detector#, show_result_pyplot
# import mmcv
# import cv2
# import torch

# # Load model
# if torch.cuda.is_available():
#     config_file = 'config_dir/cascade_mask_gpu.py'
#     checkpoint_file = 'config_dir/final.pth'
#     model = init_detector(config_file, checkpoint_file, device='cuda:0')
# else:
#     config_file = 'config_dir/cascade_mask.py'
#     checkpoint_file = 'config_dir/xxx.pth'
#     model = init_detector(config_file, checkpoint_file, device='cpu')

# # Test a single image 
# # imgs = os.listdir("imgs")
# # for img in imgs:

# def cascade_mmdet(image):
#     factor = 1.5
#     # image = os.path.join("imgs",img)
#     # img = cv2.resize(img, (int(img.shape[1]*factor),int(img.shape[0]*factor)))
#     # file_path = "static/img/"+filename
#     # img = cv2.imread(file_path)
#     # imgh, imgw, _ = img.shape
#     img = image.copy()
#     result = inference_detector(model, img)
#     # imgg = cv2.imread(image)
#     # show_result_pyplot(model, image, result, score_thr=0.85)

#     res_borderTable = []
#     res_borderlessTable = []
#     res_cell = []    
#     ## for tables with borders
#     for r in result[0][0]:
#         if r[4]>.55:
#             res_borderTable.append(r[:4].astype(int))
#         ## for cells
#     for r in result[0][1]:
#         if r[4]>.85:
#             r[4] = r[4]*100
#             res_cell.append(r.astype(int))
#         ## for borderless tables
#     for r in result[0][2]:
#         if r[4]>.55:
#             res_borderlessTable.append(r[:4].astype(int))
#     ii = 0
#     tables = res_borderTable + res_borderlessTable # x0. y0, x1, y1
#     for table in tables:
#         cv2.rectangle(img, (table[0],table[1]), (table[2],table[3]), (0, 0, 255), 3) 
#         # new_img = imgg[table[1]:table[3], table[0]:table[2]]
#         # cv2.imwrite(os.path.join("out",img.split('.')[0]+f"_{ii}_.jpg"), new_img)
#         # ii = ii + 1        
#     # cv2.imwrite(file_path+"_M.png", img)

#     return tables
