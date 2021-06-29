import matplotlib.pyplot as plt
import networkx as nx
G = nx.DiGraph()

G.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])

# nx.draw(G=G, with_labels=True, font_weight='bold')
nx.draw_circular(G=G, with_labels=True, font_weight='bold')
plt.show()
