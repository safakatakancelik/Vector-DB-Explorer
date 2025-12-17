import pandas as pd
import plotly.graph_objects as go
import umap
import numpy as np
from typing import List, Optional, Dict

class ExplorerVisualizer:
    def __init__(self):
        # UMAP reducer
        self.reducer = umap.UMAP(n_components=2, random_state=42)

    def create_plot(
        self, 
        vectors: List[List[float]], 
        documents: List[str], 
        metadatas: Optional[List[Dict]] = None,
        query_vector: Optional[List[float]] = None,
        search_result_indices: Optional[List[int]] = None
    ):
        """
        Generates a Plotly Figure with edges connecting query to results.
        Includes truncated raw vectors in hover.
        """
        if vectors is None or len(vectors) == 0:
            return "<div style='padding:20px'>No vectors to visualize.</div>"

        all_vectors = np.array(vectors)
        all_docs = list(documents)
        types = ['Document'] * len(vectors)
        
        # Handle Small Data Fallback
        if len(all_vectors) < 5:
             # Not enough data for UMAP meaningful reduction
             if all_vectors.shape[1] >= 2:
                 doc_embedding = all_vectors[:, :2]
             else:
                 doc_embedding = np.random.rand(len(all_vectors), 2)
             
             # Mock transform for query in fallback
             if query_vector is not None:
                 if len(query_vector) >= 2:
                     q_func = np.array([query_vector[:2]])
                 else:
                     q_func = np.random.rand(1, 2)
        else:
            # UMAP: Fit on Documents ONLY for stability
            # This ensures the map doesn't change when the query changes
            doc_embedding = self.reducer.fit_transform(all_vectors)

        embedding = doc_embedding

        # Append Query Project properly
        query_idx = -1
        if query_vector is not None:
            query_idx = len(embedding)
            
            # If we used UMAP, we use transform() for the query
            if len(all_vectors) >= 5:
                 q_embedding = self.reducer.transform(np.array([query_vector]))
            else:
                 # Fallback calculated above
                 q_embedding = q_func
            
            embedding = np.vstack([embedding, q_embedding])
            all_docs.append("ACTIVE QUERY")
            types.append('Query')

        # Prepare labels and hover text
        hover_texts = []
        for i, d in enumerate(all_docs):
            # Resolve raw vector source
            if types[i] == 'Query' and query_vector is not None:
                vec_source = query_vector
            else:
                vec_source = all_vectors[i]

            # Truncate raw vector for display
            raw_v_str = str(vec_source[:5]) + "..." # Show first 5 dims
            snippet = d[:80] + "..." if len(d) > 80 else d
            
            # Get projected coords
            x_coord = f"{embedding[i][0]:.2f}"
            y_coord = f"{embedding[i][1]:.2f}"
            
            label = "ACTIVE QUERY" if types[i] == 'Query' else "Doc"
            
            hover_texts.append(
                f"<b>{label}</b><br>"
                f"Coords: ({x_coord}, {y_coord})<br>"
                f"<i>{snippet}</i><br>"
                f"<span style='font-size:0.8em; color:#888'>Vec (first 5 dims): {raw_v_str}</span>"
            )

        # Build Figure manually with Graph Objects to support Edges
        fig = go.Figure()

        # 1. Draw Edges (Lines) from Query to Results
        if query_vector is not None and search_result_indices:
            q_x, q_y = embedding[query_idx]
            
            edge_x = []
            edge_y = []
            
            for res_idx in search_result_indices:
                # Ensure index is valid
                if res_idx < len(embedding):
                    r_x, r_y = embedding[res_idx]
                    edge_x.extend([q_x, r_x, None])
                    edge_y.extend([q_y, r_y, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#00ff66', dash='dot'), # Success green dashed lines
                hoverinfo='none',
                mode='lines',
                name='Matches'
            ))

        # 2. Draw Nodes
        # Split by type for coloring
        df = pd.DataFrame(embedding, columns=['x', 'y'])
        df['text'] = hover_texts
        df['type'] = types
        
        # Document Nodes
        doc_mask = [t == 'Document' for t in types]
        
        # Highlighted Documents (Search Results)
        if search_result_indices:
             # Create a specific mask/color for results? 
             # Simpler: Just rely on lines, OR recolor them.
             # Let's recolor search results to Success Green
             colors = ['#00f7ff'] * len(df) # Default Blue
             sizes = [10] * len(df)
             
             for idx in search_result_indices:
                 if idx < len(colors):
                     colors[idx] = '#00ff66' # Green
                     sizes[idx] = 14
             
             if query_idx != -1:
                 colors[query_idx] = '#ff0055' # Red/Pink for Query
                 sizes[query_idx] = 16
        else:
             colors = ['#00f7ff' if t == 'Document' else '#ff0055' for t in types]
             sizes = [10 if t == 'Document' else 16 for t in types]

        fig.add_trace(go.Scatter(
            x=df['x'], y=df['y'],
            mode='markers',
            marker=dict(size=sizes, color=colors, line=dict(width=1, color='white')),
            text=df['text'],
            hoverinfo='text',
            name='Vectors'
        ))

        # Calculate bounds
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        # Calculate centroids
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Determine strict range: Max of (Data Range + 20%) OR (Fixed 10 units)
        # This ensures that if data is tight, we zoom out to show 10 units range.
        # If data is spread out > 10, we show all data + padding.
        half_x = max(5.0, (x_max - x_min) * 0.6) # 0.5 is half, 0.6 adds 20% padding
        half_y = max(5.0, (y_max - y_min) * 0.6)
        
        # Styling
        fig.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            margin=dict(l=0, r=0, t=30, b=0),
            title=dict(text="", font=dict(size=14, color="#00f7ff")),
            xaxis=dict(showgrid=True, gridcolor='#111', range=[x_center - half_x, x_center + half_x]),
            yaxis=dict(showgrid=True, gridcolor='#111', range=[y_center - half_y, y_center + half_y]),
            showlegend=False,
            dragmode='pan' # Default interaction
        )

        return fig.to_html(
            include_plotlyjs='cdn', 
            full_html=False, 
            config={
                'displayModeBar': True, 
                'scrollZoom': True,
                'displaylogo': False
            }
        )
