class AlignmentFileManager:
    def __init__(self, input_path, file_format, out_dir=None):
        self.__out_dir = self.__check_outdir(out_dir, input_path)
        self.__file_format = self.__check_file_format(file_format)
        self.__alignment_file = self.__check_input_path(input_path)
        self.__index_file = self.__check_index_path(None)
        
    @property
    def file_format(self):
        return self.__file_format
        
    @property
    def alignment_file(self):
        return self.__alignment_file
    
    @property
    def alignment_path(self):
        return self.alignment_file.filename.decode('UTF-8')
    
    @property
    def index_file(self):
        return self.__index_file
    
    @property
    def out_dir(self):
        return self.__out_dir
    
    def __check_outdir(self, directory, input_path):
        if directory is None:
            return os.getcwd()
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'No such file or directory: {directory}')
        return directory
        
    
    def __check_file_format(self, file_format):
        supported_formats = {'sam', 'bam'}
        if file_format not in supported_formats:
            raise ValueError(f"{file_format} is not supported, sam and bam are available")
        return file_format
    
    def __check_input_path(self, input_path):
        if self.file_format == 'bam':
            return pysam.pysam.AlignmentFile(input_path, "rb")
        if self.file_format == 'sam':
            filename, file_extension = os.path.splitext(input_path)
            bam_filename = f"{filename.split('/')[-1]}.bam"
            bam_path = os.path.join(self.out_dir, bam_filename)
            if not os.path.isfile(bam_path):
                warnings.warn("try converting to BAM format...")
                pysam.sort("-o", bam_path, input_path)
            warnings.warn("found BAM file")
            return pysam.pysam.AlignmentFile(bam_path, "r")
        
    #TODO: add rel index path
    def __check_index_path(self, index_path):
        filename, file_extension = os.path.splitext(self.alignment_path)
        index_filename = os.path.join(self.out_dir, f"{filename.split('/')[-1]}.bam.bai")
        if index_path is None:
            if os.path.isfile(index_filename):
                warnings.warn("index found")
                return index_filename
            warnings.warn("Searching failed. indexing input file...")
            pysam.index(self.alignment_path)
            return index_filename
        return index_path
        