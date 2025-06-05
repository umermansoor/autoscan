class PDF2ImageConversionConfig:
    NUM_THREADS = 4
    FORMAT = "png"
    USE_PDFTOCAIRO = True
    
    # DPI settings by accuracy level
    DPI_HIGH = 200     # Best quality for complex documents, tables, handwriting
    DPI_LOW = 150      # Good quality, fastest processing, lower costs
    
    @classmethod
    def get_dpi_for_accuracy(cls, accuracy: str) -> int:
        """Get the appropriate DPI setting for the given accuracy level."""
        if accuracy == "high":
            return cls.DPI_HIGH
        else:  # "low", "medium"
            return cls.DPI_LOW
