from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        
        self.ntest = float("inf")
        self.results_dir = './results/'
        self.aspect_ratio = 1.0
        self.phase = 'test'
        self.which_epoch = 'latest'
        self.how_many = 50
        self.cluster_path = 'features_clustered_010.npy'
        self.use_encoded_image = False
        self.export_onnx = None
        self.engine = None
        self.onnx = None
        self.Arc_path = './noise_layers/simswap/arcface_checkpoint.tar'
        self.pic_a_path = './crop_224/gdg.jpg'
        self.pic_b_path = './crop_224/zrf.jpg'
        self.pic_specific_path = './crop_224/zrf.jpg'
        self.multisepcific_dir = './demo_file/multispecific'
        self.video_path = './demo_file/multi_people_1080p.mp4'
        self.temp_path = './temp'
        self.output_path = './output/'
        self.id_thres = 0.03
        self.no_simswaplogo = False
        self.use_mask = False
        self.crop_size = 224

        self.isTrain = False
