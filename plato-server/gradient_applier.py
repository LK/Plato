def start_applier(global_model, gradient_queue, optimizer):
	optimizer.zero_grad()
	local_params = gradient_queue.pop()
	for (local_param, global_param) in zip(local_params, global_model.parameters()):
		global_param.grad.data = local_param.grad.data
	optimizer.step()