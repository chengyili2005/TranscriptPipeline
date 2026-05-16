# Usage: Run this script directly within Datavyu on an opened spreadsheet.
# - Note: Specify your desired output file path and target column in the variables below.

require 'Datavyu_API.rb'
require 'json' 
require 'csv'

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Set the target column for JSON export
target_json_column = 'parent_transcript_edit'

# Set the EXACT path and filename where you want the JSON saved
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
      'start' => (cell.onset/1000.0).to_f,
      'end' => (cell.offset/1000.0).to_f,
      'text' => text_val.to_s
    }
  end

puts "Exporting current file to json"
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