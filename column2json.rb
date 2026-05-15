# Usage: Run this script directly within Datavyu on an opened spreadsheet.
# - Note: Specify your desired output file path and target column in the variables below.

require 'Datavyu_API.rb'
require 'json' 

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Set the target column for JSON export
target_json_column = 'parent_transcript_edit'

# Set the EXACT path and filename where you want the JSON saved
# e.g., "~/Desktop/my_transcript_export.json" or "C:/Users/Name/Documents/export.json"
output_path = File.expand_path("~/Desktop/#{target_json_column}_export.json")

# ---------------------------------------------------------
# EXPORT LOGIC
# ---------------------------------------------------------

puts "Starting JSON export for current file..."

# Check if the target column exists in the currently active Datavyu file
if get_column_list.include?(target_json_column)
  col = get_column(target_json_column)
  
  # Map valid cells into the required JSON hash structure
  json_data = col.cells.reject { |cell| cell.ordinal.zero? }.map do |cell|
    # Extracts text from the very first argument defined in the column.
    text_val = cell.get_code(cell.arglist.first) || ""
    
    {
      'start' => cell.onset,
      'end' => cell.offset,
      'text' => text_val.to_s
    }
  end

  # Create the directory if it doesn't exist# Usage: Run this script on a datavyu file within "input" directory (the directory of all the datavyu files you want to export).
# - Note: This input directory must be on your desktop

require 'Datavyu_API.rb'
require 'csv'
require 'json' 

# Set the folder containing the .opf files to import
dv_filedir = File.expand_path("~/Desktop/input/") + "/"
dv_filenames = Dir.entries(dv_filedir).select { |file| file.end_with?(".opf") && !file.start_with?('.') }

# Set folder to hold the new DV files being created
output_folder = File.expand_path("~/Desktop/output/") + "/"
Dir.mkdir(output_folder) unless Dir.exist?(output_folder)

# Set the target column for JSON export
target_json_column = 'parent_transcript_edit'

# Loop through each Datavyu file in the current folder
dv_filenames.each do |dv_file|
  if dv_file.include?(".opf") && dv_file[0].chr != '.'
    
    # Load the Datavyu file
    puts "Opening Datavyu file: " + dv_file
    $db, $pj = load_db(File.join(dv_filedir, dv_file))

    # Export target json column
    if get_column_list.include?(target_json_column)
      col = get_column(target_json_column)
      
      # Map valid cells into the required hash structure
      json_data = col.cells.reject { |cell| cell.ordinal.zero? }.map do |cell|
        # Assuming the text data is stored in the first argument of the column.
        # If it is stored in a specifically named argument (e.g., 'text'), replace `cell.arglist.first` with `'text'`
        text_val = cell.get_code(cell.arglist.first) || ""
        
        {
          'start' => cell.onset,
          'end' => cell.offset,
          'text' => text_val.to_s
        }
      end

      # Write JSON data to file
      json_output_file = File.join(output_folder, "#{File.basename(dv_file, '.opf')}_#{target_json_column}.json")
      File.open(json_output_file, 'w') do |f|
        f.write(JSON.pretty_generate(json_data))
      end
      puts "Successfully exported JSON for column: #{target_json_column}"
    else
      puts "Notice: Column '#{target_json_column}' not found in #{dv_file}. Skipping JSON export."
    end
    puts 'Finished processing file.'
  end
end

puts "All files successfully exported!"
  output_dir = File.dirname(output_path)
  Dir.mkdir(output_dir) unless Dir.exist?(output_dir)

  # Write JSON data to the specified file path
  File.open(output_path, 'w') do |f|
    f.write(JSON.pretty_generate(json_data))
  end
  
  puts "Successfully exported JSON for column '#{target_json_column}'."
  puts "Saved to: #{output_path}"
else
  puts "Notice: Column '#{target_json_column}' not found in the current opened file. Skipping JSON export."
end

puts "Script finished!"