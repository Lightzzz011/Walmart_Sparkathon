# Running this Python file downloads the dataset from Roboflow to your current directory.


from roboflow import Roboflow
# rf = Roboflow(api_key="qjdxqLE7RfyfFDPFipWD") # api key is shared within team
# # Add second, merge etc.: Check for overlapping images to prevent bias towards a certain image
# project = rf.workspace("furkan-bakkal").project("shopping-cart-1r48s")
# version = project.version(1)
# dataset = version.download("yolov8")

rf = Roboflow(api_key="w0YwJTLtEtGUjmOWFZ31")
project = rf.workspace("kirill-kornilov-kn3yx").project("shopping-trolley-kn5tj")
version = project.version(5)
dataset = version.download("yolov8")
