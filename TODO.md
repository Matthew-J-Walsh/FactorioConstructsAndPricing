# TODO list

This file is a todo list of things with various levels of importance. List is in no particular order.




# Speedups

## Different solvers?

## Calculation orders

Should be: make an array and fill it rather than concat?
np.einsum vs np.kron
Transposing stuff outside function
Combining beacon operations together?




# Misc improvements

## Rail level electricity transport uses length

currently it still uses area (kinda)

## Make sure transport cost stuff (rails etc.) is in active list??

it is nessiary, what to do if i need to downgrade tech?

## Robot energy costs

https://lua-api.factorio.com/stable/prototypes/LogisticRobotPrototype.html

also likely make them scaling because that makes more sense




# Transport refactoring 2.0

## Inserter proper throughput

Including known tech

## Energy costs of transports

## Space cost of transports

## Remaining transport cost pairs





# Expansion Stuff

## Expansion Beacon setups

What updates have to be done to handle diminishing beacons?



# Progression accuracy

## Quality in science factories!

How do we handle more optimal processing in science factories (which may include running items through quality for material factories)?

WE CAN COMBINE A SCIENCE AT MATERIAL FACTORY INTO ONE. The material factory won't build the science factory but that allows for both to work.

## Better catalyst managment

Find a way of handling containers and catalysts properly. Also improve catalyst classification if possible (should iron really be a catalyst?... if not, why not?)
Idea: Catalysts are only catalysts when doing their catalytic process, for example: iron is only catalytic when being used in nuclear fuel. Best method to do this would be to split the catalyst cost off individual constructs, instead detect all catalytic loops and price each for the minimal amount of that each that exists. Catalysts only matter for priced indicies (which effect factory chain addition) and pricing in standard model (not space based models).
Only issue with this is deciding what construct NEEDS the catalytic element made to be allowed (otherwise unpriced costs could end up in our model) (idea: give each item an number, all energy sources have value of 0, otherwise an item has a value equal to 1+ the minimum value of the maximums of making it. The process with the maximum value is the 'final' process)
Doesn't work, breaks linearity




# Requires expansion api

## Surface Restrictions

Properly assign the surface attribute on uncompiledconstructs.

## Productivity researches for recipe

https://www.factorio.com/blog/post/fff-376

## Expansion Decay constructs

Idea for decay: constructs with no cost that decays a decaying entity into its output.

## Fourth effect: Quality

"So, recycling an iron plate will just return an iron plate, with 25% chance"
Implement quality modules.