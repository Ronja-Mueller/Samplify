macro "Batch Analyze TIFF and Save Results" {
    // Prompt the user to select the folder containing TIFF files
    dir = getDirectory("Choose a Directory");
    
    // Create a new subfolder named 'TXT' for the results
    txtFolder = dir + "TXT/";
    File.makeDirectory(txtFolder);
    
    // Get a list of all TIFF files in the directory
    fileList = getFileList(dir);
    
    // Loop through each file in the directory
    for (i = 0; i < fileList.length; i++) {
        filename = fileList[i];
        
        // Process only TIFF files
        if (endsWith(filename, ".tif") || endsWith(filename, ".tiff")) {
            // Open the image
            open(dir + filename);
            
            // Run "Measure" command (requires pre-set ROIs or global measurement settings)
            run("Measure");
            
            // Save the Results table in the TXT folder
            resultTableName = replace(filename, ".tif", ".txt");
            resultTableName = replace(resultTableName, ".tiff", ".txt");
            saveAs("Results", txtFolder + resultTableName);
            
            // Clear the Results table
            selectWindow("Results");
            run("Clear Results");
        }
    }
    
    // Close all images and results without confirmation
    run("Close All");

    // Notify when the batch process is complete
    print("Batch analysis complete.");
}
