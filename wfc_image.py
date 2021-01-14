import copy
import math
import os
import random
import time
import cv2
import numpy as np

class Calcs:
	total_calcs1 = 0
	total_calcs2 = 0

class WaveFunction:
	def __init__(self):
		self.cells = None
		self.adjacency_matrix = None

	# extract cells
	def extract_cells(self, input_image, cell_size=3, show_work=False):
		height, width, _ = input_image.shape
		cells = {}
		cell_histogram = {}
		index = 0
		for i in range(height+1-cell_size):
			for j in range(width+1-cell_size):
				cell = input_image[i:i+cell_size,j:j+cell_size,:]
				cell_elements = tuple([val for row in cell for pixel in row for val in pixel])
				cell_frequency = cell_histogram.get(cell_elements,0) + 1
				cell_histogram[cell_elements] = cell_frequency
				if cell_frequency == 1:
					cells[index] = input_image[i:i+cell_size,j:j+cell_size,:]
					index+=1
				if show_work:
					cell = reshape_image(input_image[i:i+cell_size,j:j+cell_size,:], 100)
					cv2.imshow('cell', cell)
					cv2.waitKey(100)
		if show_work: cv2.destroyAllWindows()
		
		cell_frequencies = {}
		for ci, cell in cells.items():
			cell_elements = tuple([val for row in cell for pixel in row for val in pixel])
			cell_frequencies[ci] = cell_histogram.get(cell_elements,0)

		self.cell_histogram = cell_histogram
		self.cell_frequencies = cell_frequencies
		self.cells = cells
		self.cell_size = cell_size
	
	def cell_hash(self, cell_index):
		return tuple([val for row in self.cells[cell_index] for pixel in row for val in pixel])

	def generate_adjacency_matrix(self):
		if self.cells is None:
			print('WFC is not initialized, no cells have been extracted.')
			return False
		# based on the cell size, you can tell the max distance between cells where they can still interact:
		# how can the initial point (center?, top left?) of those points be separated? 
		# For example, cell size=5 gives -4 to 4
		max_interaction_dists = [(i,j) for i in range(-self.cell_size+1, self.cell_size) for j in range(-self.cell_size+1, self.cell_size) if (i,j) != (0,0)]
		valid_links = {}
		for ci1, cell1 in self.cells.items():
			cell1_links = {}
			for ci2, cell2 in self.cells.items():
				cell2_links = []
				for vertical_offset, horizontal_offset in max_interaction_dists:
					match = True
					for i, row in enumerate(cell1):
						Calcs.total_calcs1 += 1
						if 0 > i+vertical_offset or self.cell_size <= i+vertical_offset:
							continue	# Can't be compared due to row
						for j, pixel in enumerate(row):
							Calcs.total_calcs1 += 1
							if 0 > j+horizontal_offset or self.cell_size <= j+horizontal_offset:
								continue	# Can't be compared due to col
							if (cell1[i,j,:]!=cell2[i+vertical_offset,j+horizontal_offset,:]).any():
								match = False
								break
						if not match: break
					if match:
						cell2_links.append((vertical_offset,horizontal_offset))
				cell1_links[ci2] = cell2_links
			valid_links[ci1] = cell1_links
		# Expected usage: if offset in valid_links[ci1][ci2]: valid
		self.adjacency_matrix = valid_links


	def generate_output(self, size=(10,10), show=False):
		### initialise collapse
		if self.cells is None:
			print('WFC is not initialized, no cells have been extracted.')
			return False
		# make wave_function
		wave = np.ones((size[0]-self.cell_size+1, size[1]-self.cell_size+1, len(self.cells)))
		collapsed_wave = self.wave_function_collapse(wave, show)
		if show:
			cv2.destroyAllWindows()
		return collapsed_wave


	def wave_function_collapse(self, wave, show):
		entropy = np.sum(wave, axis=2)
		
		# if wave not viable:
		if np.min(entropy) < 1:
			# print(f'Entropy too low:\n{entropy}\n')
			#BAD
			return None

		# if the wave function has completely collapsed
		finished_cells_value = len(self.cells)+1
		min_val = np.min(np.where(entropy>1, entropy, finished_cells_value))
		if min_val == finished_cells_value:
			# DONE
			return wave

		### Print Debugging
		# if show:
			# wave_image = self.recreate_image(wave)
			# display_image(reshape_image(wave_image, 50), 'temp_construction',1, close=False)

		# chose one among the locations with lowest entropy
		min_indexes = [(i,j) for i, row in enumerate(entropy) for j, elem in enumerate(row) if elem  == min_val]
		observed_location_i, observed_location_j  = random.choice(min_indexes)
		
		# chose one of the possible cells at random (shuffle array and then go through linearly)
		possible_cells = [i for i, elem in enumerate(wave[observed_location_i,observed_location_j,:]) if elem != 0]
		# random.shuffle(possible_cells)
		
		cell_likelihood = [self.cell_frequencies[ci] for ci in possible_cells]
		chosen_cell = random.choices(possible_cells, weights=cell_likelihood)

		# Other cells are no longer valid at this location
		wave[observed_location_i,observed_location_j,:] = 0
		# This cell has been chosen
		wave[observed_location_i,observed_location_j,chosen_cell] = 1

		# Propagate
		self.propagate_consequences(wave, (observed_location_i, observed_location_j))

		wave = self.wave_function_collapse(wave, show)

		if wave is not None:
			return wave


	def wave_function_collapse_old(self, original_wave, show):
		entropy = np.sum(original_wave, axis=2)
		
		# if wave not viable:
		if np.min(entropy) < 1:
			# print(f'Entropy too low:\n{entropy}\n')
			#BAD
			return None

		# if the wave function has completely collapsed
		finished_cells_value = len(self.cells)+1
		min_val = np.min(np.where(entropy>1, entropy, finished_cells_value))
		if min_val == finished_cells_value:
			# DONE
			return original_wave

		### Print Debugging
		# if show:
			# wave_image = self.recreate_image(original_wave)
			# display_image(reshape_image(wave_image, 50), 'temp_construction',1, close=False)

		# chose one among the locations with lowest entropy
		min_indexes = [(i,j) for i, row in enumerate(entropy) for j, elem in enumerate(row) if elem  == min_val]
		observed_location_i, observed_location_j  = random.choice(min_indexes)
		
		# chose one of the possible cells at random (shuffle array and then go through linearly)
		possible_cells = [i for i, elem in enumerate(original_wave[observed_location_i,observed_location_j,:]) if elem != 0]
		random.shuffle(possible_cells)

		#solve recursively
		for attempts, chosen_cell in enumerate(possible_cells):
			wave = copy.deepcopy(original_wave)
			if attempts > 1000:
				print('taking too long')
				return None
			# Other cells are no longer valid at this location
			wave[observed_location_i,observed_location_j,:] = 0
			# This cell has been chosen
			wave[observed_location_i,observed_location_j,chosen_cell] = 1

			# Propagate
			self.propagate_consequences(wave, (observed_location_i, observed_location_j))

			wave = self.wave_function_collapse(wave, show)

			if wave is not None:
				return wave


	def propagate_consequences(self, wave, position):
		edges_h = len(wave)
		edges_w = len(wave[0])
		
		# Start with the one modified location
		disrupted_locations = [position]
		while len(disrupted_locations) > 0:
			# Starting at the oldest location
			active_loc = disrupted_locations.pop(0)
			
			# Find each location that can be affected by the current one
			nbs = [(i,j) for i in range(active_loc[0]-self.cell_size+1, active_loc[0]+self.cell_size) for j in range(active_loc[1]-self.cell_size+1, active_loc[1]+self.cell_size) if (i,j) != active_loc and 0<=i<edges_h and 0<=j<edges_w]
			
			# Find each possible cell at this location (as an array where 0 is not possible, 1 is possible)
			possible_cells = wave[active_loc[0],active_loc[1],:]
			
			# For each of the locations that can be affected
			for nb in nbs:
				original_nb_cells = copy.deepcopy(wave[nb[0],nb[1],:])
				wave[nb[0],nb[1],:] = 0

				# Look at each of the cells that I can have
				for cell_index, possible_cell in enumerate(possible_cells):
					if possible_cell == 0:
						continue # This cell is already not possible, skip!
					for adj_index, possible_adjacency in enumerate(wave[nb[0],nb[1],:]):
						Calcs.total_calcs2 += 1
						if possible_adjacency == 1 or original_nb_cells[adj_index] == 0:
							continue # This cell is either already possible, or already rejected. Either way, skip!
						if (nb[0]-active_loc[0], nb[1]-active_loc[1]) in self.adjacency_matrix[cell_index][adj_index]:
							# This is a valid match, with this offset, add it!
							wave[nb[0],nb[1],adj_index] = 1

				# If there's been changes:
				if (original_nb_cells!=wave[nb[0],nb[1],:]).any():
					# Add the modified nbs to the list of locations to check
					disrupted_locations.append(nb)


	def recreate_image(self, wave):
		image = np.ones((len(wave)+2*self.cell_size, len(wave[0])+2*self.cell_size, 3), dtype=np.uint8)
		for i in range(0, len(wave), self.cell_size):
			for j in range(0, len(wave[0]), self.cell_size):
				valid_cell_indexes = [cell_index for cell_index, val in enumerate(wave[i,j,:]) if val != 0]
				if len(valid_cell_indexes) == 1:
					gen_cell = self.cells[valid_cell_indexes[0]]
				elif len(valid_cell_indexes) == 0:
					gen_cell = np.ones((self.cell_size, self.cell_size,3)) * 50
				else:
					gen_cell = np.zeros((self.cell_size, self.cell_size,3))
					for cell_index in valid_cell_indexes:
						gen_cell = gen_cell + self.cells[cell_index]
					# print(f'Pre div:\n{gen_cell}\n')
					gen_cell = (3*gen_cell)//(len(valid_cell_indexes)*4)
					# print(f'Post div:\n{gen_cell}\n')
					
				image[i:i+self.cell_size,j:j+self.cell_size,:] = gen_cell
		image = image[:len(wave)+self.cell_size-1, :len(wave[0])+self.cell_size-1,:]
		return image

	def debug(self, delay=150, scaling=100):
		for cell in self.cells.values():
			display_image(reshape_image(cell, scaling), delay=delay, close=False)
		# print('Press any key to continue')
		
		# cell1 in the center, cell2 offset, 
		for ci1, cell_links in self.adjacency_matrix.items():
			cell1_image = np.ones((self.cell_size*6, self.cell_size*5, 3), dtype=np.uint8)
			cell1_image = cell1_image*50
			cell1_image[self.cell_size*2:self.cell_size*3,self.cell_size*2:self.cell_size*3,:] = self.cells[ci1]
			cell1_image[self.cell_size*5-1:self.cell_size*6-1,1:self.cell_size+1,:] = self.cells[ci1]
			for ci2, valid_offsets in cell_links.items():
				cell2_image = copy.deepcopy(cell1_image)
				cell2_image[self.cell_size*5-1:self.cell_size*6-1,-self.cell_size-1:-1,:] = self.cells[ci2]
				for offset in valid_offsets:
					# vertical_offset, horizontal_offset
					new_image = copy.deepcopy(cell2_image)
					print(ci1, ci2, offset)
					new_image[self.cell_size*2-offset[0]:self.cell_size*3-offset[0],self.cell_size*2-offset[1]:self.cell_size*3-offset[1],:] = self.cells[ci2]
					display_image(reshape_image(new_image, scaling), delay=delay, close=False)

		cv2.waitKey(1)
		cv2.destroyAllWindows()

	def print_wave(self, wave):
		for row in wave:
			for cells in row:
				for index, elem in enumerate(cells):
					if elem !=0:
						print(index, end='\t')
			print()

	def save_cells(self, img_name='extracted_cells.png'):
		cells_per_row = math.ceil(math.sqrt(len(self.cells)))
		
		cell_image = np.ones((math.ceil(len(self.cells)/cells_per_row)*(self.cell_size+1)+1, cells_per_row*(self.cell_size+1)+1, 3), dtype=np.uint8)
		cell_image = cell_image * 50
		
		for ci, cell in self.cells.items():
			row = ci // cells_per_row 
			col = ci % cells_per_row
			cell_image[1+row*(self.cell_size+1):(row+1)*(self.cell_size+1),1+col*(self.cell_size+1):(col+1)*(self.cell_size+1),:] = cell

		save_image(reshape_image(cell_image, 20), img_name)

		# cell1_image = cell1_image*50
		# for row in wave:
			# for cells in row:
				# for index, elem in enumerate(cells):
					# if elem !=0:
						# print(index, end='\t')
			# print()



# WFC for images

### Steps:
# [x] read input_image
# [x] extract cells
# [x] generate wavefunction
# [x] collapse wavefunction
# [x] generate output_image
# [x] display output_image
# [x] save output_image

# read input_image
def read_image(image_path, flag=-1):
	# flag
	# cv::IMREAD_UNCHANGED = -1,
	# cv::IMREAD_GRAYSCALE = 0,
	# cv::IMREAD_COLOR = 1,
	return  cv2.imread(image_path, flag)
	### Not needed:
	# return  cv2.cvtColor(cv2.imread(image_path, flag), cv2.COLOR_RGB2BGR)

# display output_image
def display_image(image, img_name='Temp Image', delay=0, close=True):
	cv2.imshow(img_name, image)
	cv2.waitKey(delay)
	if close: cv2.destroyAllWindows()

# save output_image
def save_image(image, img_name='temp_image.png', save_location=None):
	if save_location == None:
		save_location = f"{os.environ['DEVICE_IDENTIFIER']}\\Desktop\\Temp\\wfc"
	print(f'Saving image to {save_location}\\{img_name}')
	cv2.imwrite(f'{save_location}\\{img_name}', image)

def reshape_image(image, scaling_factor):
	return np.kron(image, np.ones((scaling_factor, scaling_factor, 1))).astype(np.uint8)


def main():
	image_path = 'images/base_image5.png'
	image_path = 'sample_jellyfish.png'
	input_image = read_image(image_path)
	big_image = reshape_image(input_image, 50)
	# display_image(big_image, 'input_image', close=False)

	# display_image(reshape_image(
	# [[[255,255,255], [255,255,1], [255,255,0]],
	# [[1,1,255], [1,1,1], [1,1,0]],
	# [[0,0,255], [0,0,1], [0,0,0]]]
	# , 40), 'wave_image')
	# exit(0)
	attempts = 5
	
	wave_fct = WaveFunction()
	wave_fct.extract_cells(input_image, 2, show_work=False)
	wave_fct.generate_adjacency_matrix()
	
	# wave_fct.debug(delay=1, scaling=50)
	# exit()
	
	for _ in range(attempts):
		wave_image = wave_fct.generate_output((12,12), show=True)
		if wave_image is None:
			print('Didn\'t work')
		else:
			break
	
	wave_fct.save_cells()
	wave_fct.print_wave(wave_image)
	# exit()
	wave_image = wave_fct.recreate_image(wave_image)
	display_image(reshape_image(wave_image, 50), 'wave_image')
	
	# test_wave=[[1,2,3],[4,5,6],[7,8,9]]
	# propagate_consequences(test_wave, (1,1))
	# propagate_consequences(test_wave, (0,0))
	# propagate_consequences(test_wave, (1,2))

if __name__ == "__main__":
	main()
	print(f'total_calcs:\n\t{Calcs.total_calcs1}\n\t{Calcs.total_calcs2}')