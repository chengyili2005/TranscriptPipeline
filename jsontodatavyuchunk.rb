# Usage: Open your spreadsheet inside Datavyu. Run this script.
# It will prompt you to select a JSON file and safely import it using optimized chunk sizes.
require 'Datavyu_API.rb'
require 'json'

begin
  # Import Java GUI classes for the file picker window
  java_import javax.swing.JFileChooser
  java_import javax.swing.filechooser.FileNameExtensionFilter
  java_import javax.swing.JFrame
  java_import javax.swing.JOptionPane

  # ---------------------------------------------------------
  # CONFIGURATION
  # ---------------------------------------------------------
  target_column_name = 'mfa_finished_parent_transcript_edit'
  CHUNK_SIZE = 50 # Adjust batch size based on system performance

  # ---------------------------------------------------------
  # GUI FILE CHOOSER SETUP
  # ---------------------------------------------------------
  puts "Opening file selection dialog..."
  
  # Initialize an invisible background window anchor
  frame = JFrame.new("Import JSON")
  frame.setDefaultCloseOperation(JFrame::DISPOSE_ON_CLOSE)
  frame.setSize(200, 200)
  frame.setLocationRelativeTo(nil)
  
  # Setup the file chooser properties
  jfc = JFileChooser.new
  jfc.setAcceptAllFileFilterUsed(false)
  jfc.setMultiSelectionEnabled(false)
  jfc.setDialogTitle('Select JSON transcript file to import')
  
  # Restrict selection strictly to .json files
  extensions = ["json"].to_java(:String)
  filter = FileNameExtensionFilter.new("JSON Transcripts (*.json)", extensions)
  jfc.addChoosableFileFilter(filter)
  
  # Display the dialog box to the user
  frame.setVisible(true)
  result = jfc.showOpenDialog(frame)
  frame.dispose # Safely close the window tracker
  
  # Check if the user canceled the prompt
  if result != JFileChooser::APPROVE_OPTION
    puts "Selection canceled by user. Aborting."
    return
  end
  
  # Extract the clean file path from the dialog picker
  file_path = jfc.getSelectedFile.getPath
  puts "Selected File: #{file_path}"

  # ---------------------------------------------------------
  # CHUNKED IMPORT LOGIC
  # ---------------------------------------------------------
  # Parse the selected JSON file
  json_payload = File.read(file_path)
  entries = JSON.parse(json_payload)

  total_entries = entries.size
  total_chunks = (total_entries.to_f / CHUNK_SIZE).ceil
  puts "Processing #{total_entries} entries in #{total_chunks} chunks..."

  # Build the target column structure with 'text' as its internal text argument
  new_col = new_column(target_column_name, 'text')

  # Process the array in smaller sliced pieces
  (0...total_chunks).each do |chunk_idx|
    start_idx = chunk_idx * CHUNK_SIZE
    end_idx = [start_idx + CHUNK_SIZE, total_entries].min
    
    chunk_slice = entries[start_idx...end_idx]
    puts "Processing chunk #{chunk_idx + 1}/#{total_chunks} (cells #{start_idx + 1}-#{end_idx})..."

    chunk_slice.each do |entry|
      # Convert decimal seconds back into Datavyu integer milliseconds
      onset_ms  = ((entry['start'] || 0) * 1000).to_i
      offset_ms = ((entry['end'] || 0) * 1000).to_i
      text_data = entry['text'] || ""

      # Generate cell container inside our chunk structure
      cell = new_col.make_new_cell()
      cell.change_code("onset", onset_ms)
      cell.change_code("offset", offset_ms)
      cell.change_code("text", text_data.to_s)
    end
    
    # Optional: A brief, tiny pause can help give JRuby's garbage collector 
    # a window to clear memory during massive file structures
    sleep(0.01) 
  end

  # Commit the fully built container back onto the visible GUI matrix
  puts "Finalizing injection into active Datavyu spreadsheet window..."
  set_column(target_column_name, new_col)
  
  puts "Successfully imported data to column: '#{target_column_name}'"
  JOptionPane.showMessageDialog(nil, "Import completed successfully!\nProcessed #{total_entries} cells.", "Success", JOptionPane::INFORMATION_MESSAGE)

rescue => e
  puts "Error occurred: #{e.message}"
  JOptionPane.showMessageDialog(nil, "Error: #{e.message}", "Import Error", JOptionPane::ERROR_MESSAGE)
end