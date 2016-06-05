cdef:
	float* c_get_state() 
	int c_do_action(int action) except -1
	float c_get_score()
	int c_get_time()
	int c_is_level_finished()