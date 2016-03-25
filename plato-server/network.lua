require 'torch'
require 'nn'
require 'optim'
require 'hdf5'
local MongoClient = require("mongorover.MongoClient")

cmd = torch.CmdLine()
cmd:option('-snapshot', '')
params = cmd:parse(arg)

local net = nil

-- Set up the network
if params.snapshot ~= '' then
  net = torch.load(params.snapshot)
  print('Loaded network from snapshot')
else
  net = nn.Sequential()
  net:add(nn.Linear(5, 10))
  net:add(nn.ReLU())
  net:add(nn.Linear(10, 3))
  net:add(nn.ReLU())
  net:add(nn.Linear(3, 3))
  net:add(nn.ReLU())
  net:add(nn.Linear(3, 1))
end

local crit = nn.MSECriterion()
local delayed_net = net:clone() -- We use this to generate our Q-targets

local weights, grad_weights = net:getParameters()

-- Connect to database
local client = MongoClient.new('mongodb://localhost:27017/')
local database = client:getDatabase('plato')
local collection = database:getCollection('games')

local steps = 0

-- Take a state (from Mongo) and return an array
local function state_to_array(state)
  return {state.heading, state.energy, state.oppBearing, state.oppEnergy, state.action}
end

local function f(w)

  -- Sample minibatch of 64 games
  local mb_x = torch.Tensor(64, 5)
  local mb_y = torch.Tensor(64, 1)

  local games = collection:aggregate({
    {
      ['$sample'] = {size = 64}
    }
  })
  local idx = 1

  for game in games do
    -- Sample a random state from each game
    local state_idx = math.random(#game.history)
    local state = game.history[state_idx] -- sample random state
    mb_x[idx] = state_to_array(state)

    -- Calculate Q-target
    if state_idx == #game.history then -- is this a terminal state?
      -- If it's a terminal state, Q-target is just the reward
      if state.res == 'W' then
        mb_y[idx] = {1}
      else
        mb_y[idx] = {0}
      end
    else
      -- Otherwise, use Bellman optimality equation
      local max = 0
      local arr = state_to_array(game.history[state_idx+1])

      -- Find maximum action for following state
      for a = 1,5 do
        arr[5] = a
        max = math.max(max, delayed_net:forward(torch.Tensor(arr)))
      end
      mb_y[idx] = {max} -- gamma = 1, r = 0
    end

    idx = idx + 1
  end

  -- Forward-pass on minibatch
  local estimates = net:forward(mb_x)

  -- Compute loss with Q-targets (using MSE)
  local loss = crit:forward(estimates, mb_y)

  -- Calculate and return gradients
  net:zeroGradParameters()
  local destimates = crit:backward(estimates, mb_y)
  local dx = net:backward(mb_x, destimates)

  return loss, grad_weights

end

local steps = 0

while true do
  if steps % 10000 == 0 then
    print('Step ' .. steps)

    -- Update our "delayed" network to our current weights
    delayed_net = net:clone()

    local weights_, _ = net:parameters()
    for key, value in pairs(weights_) do print(key, value) end

    -- Dump a snapshot
    local file = hdf5.open('snapshots/' .. steps .. '.h5', 'w')
    file:write('/network', weights)
    file:close()

    torch.save('snapshots/' .. steps, net)
  end

  local state = {learningRate=1e-3}
  optim.adam(f, weights, state)

  steps = steps + 1
end
