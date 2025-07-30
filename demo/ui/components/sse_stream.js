import {
  LitElement,
  html,
} from 'https://cdn.jsdelivr.net/gh/lit/dist@3/core/lit-core.min.js';

class SSEStream extends LitElement {
  static properties = {
    url: {type: String},
    triggerEvent: {type: String},
  };

  constructor() {
    super();
    this.eventSource = null;
  }

  render() {
    return html`<div></div>`;
  }

  firstUpdated() {
    if (this.url) {
      this.startSSE();
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.stopSSE();
  }

  startSSE() {
    if (this.eventSource) {
      this.stopSSE();
    }

    this.eventSource = new EventSource(this.url);
    
    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.dispatchEvent(
          new MesopEvent(this.triggerEvent, {
            value: data,
          }),
        );
      } catch (error) {
        console.error('Error parsing SSE data:', error);
      }
    };

    this.eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      // Try to reconnect after 5 seconds
      setTimeout(() => {
        if (this.eventSource && this.eventSource.readyState === EventSource.CLOSED) {
          this.startSSE();
        }
      }, 5000);
    };

    this.eventSource.onopen = () => {
      console.log('SSE connection opened');
    };
  }

  stopSSE() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  updated(changedProperties) {
    if (changedProperties.has('url')) {
      this.startSSE();
    }
  }
}

customElements.define('sse-stream-component', SSEStream);