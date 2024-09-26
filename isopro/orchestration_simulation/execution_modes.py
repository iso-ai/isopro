from concurrent.futures import ThreadPoolExecutor

def run_in_sequence(components):
    for component in components:
        component.run()

def run_in_parallel(components):
    with ThreadPoolExecutor() as executor:
        executor.map(lambda component: component.run(), components)

def run_as_node(components):
    # Implement custom node-based execution logic here
    for component in components:
        component.run()