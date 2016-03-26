require 'torch'
require 'nn'
require 'optim'
require 'hdf5'
local MongoClient = require("mongorover.MongoClient")

cmd = torch.CmdLine()
cmd:option('-snapshot', '')
params = cmd:parse(arg)

local net = nil
local steps = 0

-- Set up the network
if params.snapshot ~= '' then
  net = torch.load('snapshots/' .. params.snapshot)
  steps = params.snapshot
  print('Loaded network from snapshot')
else
  net = nn.Sequential()
  net:add(nn.Linear(5, 256))
  net:add(nn.ReLU())
  net:add(nn.Linear(256, 256))
  net:add(nn.ReLU())
  net:add(nn.Linear(256, 256))
  net:add(nn.ReLU())
  net:add(nn.Linear(256, 1))
end

local crit = nn.MSECriterion()
local delayed_net = net:clone() -- We use this to generate our Q-targets

local weights, grad_weights = net:getParameters()

-- Connect to database
local client = MongoClient.new('mongodb://localhost:27017/')
local database = client:getDatabase('plato')
local collection = database:getCollection('games')

local steps = 0

local function dump(steps)
  local file = hdf5.open('snapshots/' .. steps .. '.h5', 'w')
  file:write('/network', weights)
  file:write('/epsilon', torch.Tensor({math.min(1/((steps+1) * 1e-4), 1.0)}))
  file:close()

  torch.save('snapshots/' .. steps, net)
end

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
    mb_x[idx] = torch.Tensor(state_to_array(state))

    -- Calculate Q-target
    if state_idx == #game.history then -- is this a terminal state?
      -- If it's a terminal state, Q-target is just the reward
      if state.res == 'W' then
        mb_y[idx] = torch.Tensor({1})
      else
        mb_y[idx] = torch.Tensor({0})
      end
    else
      -- Otherwise, use Bellman optimality equation
      local max = 0
      local arr = state_to_array(game.history[state_idx+1])

      -- Find maximum action for following state
      for a = 0,5 do
        arr[5] = a
        max = math.max(max, delayed_net:forward(torch.Tensor(arr))[1])
      end
      mb_y[idx] = torch.Tensor({max}) -- gamma = 1, r = 0
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

if steps == 0 then dump(0) end

while true do
  if collection:count() < 64 then
    local time = os.clock()
    while os.clock() - time < 5 do end
  else
    if steps % 100 == 0 then
      print('Step ' .. steps)

      -- Update our "delayed" network to our current weights
      delayed_net = net:clone()

      -- local weights_, _ = net:parameters()
      -- for key, value in pairs(weights_) do print(key, value) end

      -- Dump a snapshot
      dump(steps)
    end

    local state = {learningRate=1e-3}
    optim.adam(f, weights, state)

    steps = steps + 1
  end
end
