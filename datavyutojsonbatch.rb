# Usage: Run this script to batch process all .opf files inside a "datavyutojson" folder on your Desktop.
require 'Datavyu_API.rb'
require 'csv'
require 'json' 

# ---------------------------------------------------------
# CONFIGURATION & PATHS
# ---------------------------------------------------------
dv_filedir = File.expand_path("~/Desktop/datavyutojsoninput/") + "/"
output_folder = File.expand_path("~/Desktop/datavyutojsonoutput/") + "/"

# Create output directory if missing
Dir.mkdir(output_folder) unless Dir.exist?(output_folder)

target_json_column = 'parent_transcript_edit'

# Grab all valid .opf files
dv_filenames = Dir.entries(dv_filedir).select { |file| file.end_with?(".opf") && !file.start_with?('.') }

# ---------------------------------------------------------
# BATCH PROCESSING LOOP
# ---------------------------------------------------------
dv_filenames.each do |dv_file|
  puts "Opening Datavyu file: #{dv_file}"
  $db, $pj = load_db(File.join(dv_filedir, dv_file))

  if get_column_list.include?(target_json_column)
    col = get_column(target_json_column)
    
    json_data = col.cells.reject { |cell| cell.ordinal.zero? }.map do |cell|
      text_val = cell.get_code(cell.arglist.first) || ""
      
      {
        # Using 1000.0 fixes the truncation/rounding bug
        'start' => cell.onset / 1000.0,
        'end'   => cell.offset / 1000.0,
        'text'  => text_val.to_s
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

puts "All files successfully exported!"